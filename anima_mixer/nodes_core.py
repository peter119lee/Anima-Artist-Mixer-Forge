"""Core ComfyUI nodes: Pack, CrossAttn patcher, and the layer Probe."""

import logging


from .anchor import make_sigma_capture
from .constants import (
    COMBINE_CHOICES,
    COMBINE_CONCAT,
    COMBINE_LOWRANK_AVG,
    COMBINE_OUTPUT_AVG,
    FUSION_BASE_PRESERVE,
    FUSION_CHOICES,
    FUSION_CONCAT_WITH_BASE,
    FUSION_INTERPOLATE,
    ANCHOR_LAYER_THRESHOLD_DISABLED,
    ANCHOR_SEEDS_MAX,
    CONTRIB_BALANCE_ALPHA_DEFAULT,
    MIXED_DELTA_CAP_RATIO_DEFAULT,
    MIXED_DELTA_CAP_RATIO_MAX,
    NORM_LOCK_SCOPE_PER_ARTIST,
    NORM_LOCK_TOKEN,
    STATIC_CAPTURE_BLEND_ALPHA_DEFAULT,
    STATIC_CAPTURE_K_DEFAULT,
    STATIC_CAPTURE_K_MAX,
    STATIC_CAPTURE_MODE_OUTPUT,
)
from .options import merge_runtime_options
from .parsing import (
    build_passthrough_prompt,
    resolve_artist_layer_routes,
    resolve_artist_timing_routes,
    resolve_target_blocks_from_options,
)
from .patching import (
    describe_external_cross_attn_patches,
    extract_conditioning,
    make_cross_attn_forward_patch,
    unwrap_cross_attn,
    unwrap_cross_attn_forward,
    validate_model,
)
from .wrapper import CrossAttnWrapper

logger = logging.getLogger(__name__)


class AnyType(str):
    """Wildcard type marker accepted by ComfyUI for 'any' inputs."""

    def __ne__(self, other):
        return False


ANY_TYPE = AnyType("*")


# ComfyUI's percent_to_sigma(0.0) returns a huge sentinel (999999999.9)
# meaning "always active". Fine for window-inclusion tests, but useless as a
# finite anchor for fade interpolation.
_SIGMA_SENTINEL = 1e8


def _finite_sigma(ms, percent):
    """percent_to_sigma with a finite stand-in for the percent=0 sentinel.

    The stand-in is the real near-max sigma padded by 0.1% so the very first
    sampling step (at sigma_max) still falls inside the window.
    """
    s = float(ms.percent_to_sigma(percent))
    if s >= _SIGMA_SENTINEL:
        s = float(ms.percent_to_sigma(1e-4)) * 1.001
    return s


def _percent_window_to_sigma_route(ms, timing):
    """Convert a percent-space (start, end, fade) window to sigma space.

    Returns (lo, hi, fade_in_lo, fade_out_hi) for timing_fade_factor.
    Sigma decreases as percent increases, so window start -> hi.
    """
    start, end, fade = timing
    fade_eff = min(float(fade), max(0.0, (end - start) / 2.0))
    hi = _finite_sigma(ms, start)
    lo = _finite_sigma(ms, end)
    fade_in_lo = _finite_sigma(ms, min(start + fade_eff, end))
    fade_out_hi = _finite_sigma(ms, max(end - fade_eff, start))
    lo, hi = min(lo, hi), max(lo, hi)
    fade_in_lo = min(max(fade_in_lo, lo), hi)
    fade_out_hi = min(max(fade_out_hi, lo), hi)
    return (lo, hi, fade_in_lo, fade_out_hi)


def _build_runtime_state(
    enabled,
    fusion_mode,
    combine_mode,
    strength,
    apply_to_uncond,
    raws,
    ids_list,
    w_list,
    user_weights,
    labels,
    artist_layer_routes,
    has_artist_layer_routes,
    artist_timing_routes,
    has_artist_timing_routes,
    normalize_w,
    has_explicit_weights,
    preset_name,
    adv,
    dm,
    sigma_range,
    external_patches,
):
    return {
        "enabled": bool(enabled),
        "fusion_mode": fusion_mode,
        "combine_mode": combine_mode,
        "strength": float(strength),
        "apply_to_uncond": bool(apply_to_uncond),
        "raws": raws,
        "ids_list": ids_list,
        "w_list": w_list,
        "user_weights": user_weights,
        "labels": labels,
        "artist_layer_routes": artist_layer_routes,
        "has_artist_layer_routes": has_artist_layer_routes,
        "artist_timing_routes": artist_timing_routes,
        "has_artist_timing_routes": has_artist_timing_routes,
        "normalize_weights": normalize_w,
        "has_explicit_weights": has_explicit_weights,
        "preset_name": preset_name,
        "artist_ema_alpha": float(adv.get("artist_ema_alpha", 0.0)),
        "lowrank_k": int(adv.get("lowrank_k", 1)),
        "artist_static_capture": bool(adv.get("artist_static_capture", False)),
        "static_capture_k": int(adv.get("static_capture_k", STATIC_CAPTURE_K_DEFAULT)),
        "static_capture_mode": str(adv.get("static_capture_mode", STATIC_CAPTURE_MODE_OUTPUT)),
        "static_capture_blend_alpha": float(
            adv.get("static_capture_blend_alpha", STATIC_CAPTURE_BLEND_ALPHA_DEFAULT)
        ),
        "artist_anchor_q": bool(adv.get("artist_anchor_q", False)),
        "anchor_seeds_count": int(adv.get("anchor_seeds_count", 1)),
        "anchor_user_blend": float(adv.get("anchor_user_blend", 0.0)),
        "anchor_deep_layer_threshold": int(
            adv.get("anchor_deep_layer_threshold", ANCHOR_LAYER_THRESHOLD_DISABLED)
        ),
        "anchor_refresh_each_step": bool(adv.get("anchor_refresh_each_step", False)),
        "max_batch_artists": int(adv.get("max_batch_artists", 0) or 0),
        "artist_q_reuse": bool(adv.get("artist_q_reuse", False)),
        "low_vram_cache": bool(adv.get("low_vram_cache", False)),
        "match_base_norm": bool(adv.get("match_base_norm", False)),
        "anchor_base_norm_ref": bool(adv.get("anchor_base_norm_ref", False)),
        "norm_lock_mode": str(adv.get("norm_lock_mode", NORM_LOCK_TOKEN)),
        "norm_lock_scope": str(adv.get("norm_lock_scope", NORM_LOCK_SCOPE_PER_ARTIST)),
        "contribution_balance": bool(adv.get("contribution_balance", False)),
        "contribution_balance_alpha": float(
            adv.get("contribution_balance_alpha", CONTRIB_BALANCE_ALPHA_DEFAULT)
        ),
        "mixed_delta_cap": bool(adv.get("mixed_delta_cap", False)),
        "mixed_delta_cap_ratio": float(adv.get("mixed_delta_cap_ratio", MIXED_DELTA_CAP_RATIO_DEFAULT)),
        "individuals": None,
        "real_lens": None,
        "dm_ref": dm,
        "sigma_range": sigma_range,
        "stabilizer_min_sigma": adv.get("stabilizer_min_sigma"),
        "current_sigma": None,
        "external_cross_attn_patches": external_patches,
        "_disabled_layers": set(),
        "_run_last_sigma": None,
        "_ema_cache": {},
        "_ema_last_sigma": None,
        "_static_cache": {},
        "_static_last_sigma": None,
        "_ctx_fp_memo": {},
        "_anchor_cache": {},
        "_anchor_base_cache": {},
        "_anchor_cache_key": None,
        "_anchor_last_sigma": None,
        "_in_anchor_run": False,
        "_anchor_failed": False,
    }


class AnimaArtistCrossAttn:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "artist_pack": ("ANIMA_PACK",),
                "combine_mode": (
                    COMBINE_CHOICES,
                    {
                        "default": COMBINE_OUTPUT_AVG,
                        "tooltip": (
                            "concat: concatenate artist tokens as K/V\n"
                            "output_avg: weighted average of per-artist attention outputs\n"
                            "lowrank_avg: deterministic low-rank constraint on artist deltas"
                        ),
                    },
                ),
                "fusion_mode": (
                    FUSION_CHOICES,
                    {
                        "default": FUSION_INTERPOLATE,
                        "tooltip": (
                            "interpolate: out = lerp(base, artist, strength)\n"
                            "concat_with_base: KV=[base; artist] single forward\n"
                            "base_preserve: strip the artist component parallel to base\n"
                            "  out = base + strength * proj_perp(artist - base)\n"
                            "  base content direction stays untouched\n"
                            "  compatible with lowrank_avg / EMA"
                        ),
                    },
                ),
                "strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 4.0,
                        "step": 0.05,
                        "tooltip": (
                            "Artist injection strength.\n"
                            "0.0-1.0: interpolation lerp(base, artist, strength)\n"
                            "1.0-4.0: extrapolation base + strength * (artist - base)\n"
                            "  Amplifies the artist's deviation; decoupled from artist count.\n"
                            "  Recommended 1.5-2.5; above 3 tends to oversaturate."
                        ),
                    },
                ),
                "enabled": ("BOOLEAN", {"default": True}),
                "apply_to_uncond": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "advanced_options": ("ANIMA_OPTS",),
                "preset": (
                    "ANIMA_PRESET",
                    {
                        "tooltip": (
                            "Compatibility input for older workflows. For preset "
                            "workflows, prefer Anima Artist Apply Preset so the "
                            "manual combine/fusion/strength widgets are not shown."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING")
    RETURN_NAMES = ("model", "base_prompt")
    FUNCTION = "patch"
    CATEGORY = "Anima/Setup"

    def patch(
        self,
        model,
        artist_pack,
        combine_mode,
        fusion_mode,
        strength,
        enabled,
        apply_to_uncond,
        advanced_options=None,
        preset=None,
    ):
        base_prompt = ""
        artist_count = 0
        if isinstance(artist_pack, dict):
            base_prompt = str(artist_pack.get("base_prompt", "") or "")
            artist_count = len(artist_pack.get("labels") or [])
        combine_mode, fusion_mode, strength, adv, preset_name = merge_runtime_options(
            combine_mode,
            fusion_mode,
            strength,
            advanced_options,
            preset,
            base_prompt=base_prompt,
            artist_count=artist_count,
        )

        if not isinstance(artist_pack, dict):
            raise ValueError(
                "[AnimaCrossAttn] artist_pack has the wrong type; connect the "
                "output of an AnimaArtistPack node"
            )

        base_cond_out = artist_pack.get("base_conditioning")
        if base_cond_out is None:
            raise ValueError(
                "[AnimaCrossAttn] artist_pack is missing base_conditioning. "
                "Restart ComfyUI so AnimaArtistPack reloads at the current version"
            )

        conditionings = artist_pack.get("conditionings") or []
        if not enabled or not conditionings:
            # Nothing to inject: hand back the unpatched model with zero overhead.
            return (model, base_cond_out)

        labels = artist_pack.get("labels") or []
        parsed_weights = artist_pack.get("weights")
        if isinstance(parsed_weights, (list, tuple)) and len(parsed_weights) == len(labels):
            user_weights = [float(w) for w in parsed_weights]
        else:
            user_weights = [1.0] * len(labels)

        if adv.get("prompt_passthrough", False):
            clip = artist_pack.get("clip")
            if clip is None:
                raise ValueError(
                    "[AnimaCrossAttn] prompt_passthrough requires an artist_pack "
                    "built by the current AnimaArtistPack node."
                )
            if any(str(route or "").strip() for route in artist_pack.get("layer_routes") or []):
                raise ValueError(
                    "[AnimaCrossAttn] prompt_passthrough does not support layer routes; "
                    "use balanced or another mixer preset for @layer routing."
                )
            if any(str(route or "").strip() for route in artist_pack.get("timing_routes") or []):
                raise ValueError(
                    "[AnimaCrossAttn] prompt_passthrough does not support timing routes; "
                    "use balanced or another mixer preset for %timing routing."
                )
            prompt_text = build_passthrough_prompt(
                artist_pack.get("raw_artist_chain", ""),
                labels,
                user_weights,
                artist_pack.get("has_explicit_weights", False),
                base_prompt,
            )
            try:
                tokens = clip.tokenize(prompt_text)
                direct_conditioning = clip.encode_from_tokens_scheduled(tokens)
            except Exception as e:
                raise ValueError(
                    "[AnimaCrossAttn] prompt_passthrough failed to encode direct "
                    f"prompt (text={prompt_text!r}): {e}"
                )
            return (model, direct_conditioning)

        start_percent = float(adv.get("start_percent", 0.0))
        end_percent = float(adv.get("end_percent", 1.0))
        normalize_w = bool(adv.get("normalize_weights", True))
        artist_ema_alpha = float(adv.get("artist_ema_alpha", 0.0))
        artist_static_capture = bool(adv.get("artist_static_capture", False))
        static_capture_k = int(adv.get("static_capture_k", STATIC_CAPTURE_K_DEFAULT))
        adv["static_capture_k"] = max(1, min(static_capture_k, STATIC_CAPTURE_K_MAX))
        static_blend_alpha = float(adv.get("static_capture_blend_alpha", STATIC_CAPTURE_BLEND_ALPHA_DEFAULT))
        adv["static_capture_blend_alpha"] = max(0.0, min(1.0, static_blend_alpha))
        artist_anchor_q = bool(adv.get("artist_anchor_q", False))
        anchor_seeds_count = int(adv.get("anchor_seeds_count", 1))
        adv["anchor_seeds_count"] = max(1, min(anchor_seeds_count, ANCHOR_SEEDS_MAX))
        anchor_user_blend = float(adv.get("anchor_user_blend", 0.0))
        adv["anchor_user_blend"] = max(0.0, min(1.0, anchor_user_blend))
        contribution_balance_alpha = float(
            adv.get("contribution_balance_alpha", CONTRIB_BALANCE_ALPHA_DEFAULT)
        )
        adv["contribution_balance_alpha"] = max(0.0, min(1.0, contribution_balance_alpha))
        mixed_delta_cap_ratio = float(adv.get("mixed_delta_cap_ratio", MIXED_DELTA_CAP_RATIO_DEFAULT))
        adv["mixed_delta_cap_ratio"] = max(0.0, min(MIXED_DELTA_CAP_RATIO_MAX, mixed_delta_cap_ratio))
        stabilizer_end_percent = float(adv.get("stabilizer_end_percent", 1.0))
        adv["stabilizer_end_percent"] = max(0.0, min(1.0, stabilizer_end_percent))

        use_sigma_range = (start_percent > 0.0) or (end_percent < 1.0)
        use_stabilizer_window = adv["stabilizer_end_percent"] < 1.0
        need_sigma_capture = (
            use_sigma_range
            or (artist_ema_alpha > 0.0)
            or artist_static_capture
            or artist_anchor_q
            or use_stabilizer_window
        )

        # Mutual-exclusion checks.
        if artist_static_capture and artist_ema_alpha > 0.0:
            logger.info(
                "[AnimaCrossAttn] artist_ema_alpha=%.2f is ignored while "
                "artist_static_capture is on (artist outputs are already static).",
                artist_ema_alpha,
            )
        if artist_static_capture and fusion_mode == FUSION_CONCAT_WITH_BASE:
            logger.warning(
                "[AnimaCrossAttn] artist_static_capture is incompatible with "
                "fusion=concat_with_base (x changes every step); static capture "
                "is ignored for this run."
            )
        if artist_anchor_q and artist_static_capture:
            logger.warning(
                "[AnimaCrossAttn] artist_anchor_q and artist_static_capture are "
                "mutually exclusive; static_capture is disabled (anchor_q wins)."
            )
            adv["artist_static_capture"] = False
        if artist_anchor_q and fusion_mode == FUSION_CONCAT_WITH_BASE:
            logger.warning(
                "[AnimaCrossAttn] artist_anchor_q is incompatible with "
                "fusion=concat_with_base; anchor_q is disabled for this run."
            )
            adv["artist_anchor_q"] = False
        if not adv.get("artist_anchor_q", False):
            adv["anchor_base_norm_ref"] = False
            adv["anchor_refresh_each_step"] = False
        layer_route_texts = artist_pack.get("layer_routes") or []
        timing_route_texts = artist_pack.get("timing_routes") or []

        raws, ids_list, w_list = [], [], []
        for idx, c in enumerate(conditionings):
            raw, ids, w = extract_conditioning(c)
            if raw is None:
                label = labels[idx] if idx < len(labels) else f"#{idx}"
                raise ValueError(
                    f"[AnimaCrossAttn] artist[{label}] conditioning is empty. Do the CLIP and model match?"
                )
            raws.append(raw)
            ids_list.append(ids)
            w_list.append(w)

        n = len(raws)
        has_explicit_weights = bool(artist_pack.get("has_explicit_weights", False))
        if len(user_weights) != n:
            user_weights = [1.0] * n
            has_explicit_weights = False

        if has_explicit_weights and normalize_w:
            normalize_w = False
            logger.info("[AnimaCrossAttn] explicit ::weight detected; normalize_weights is bypassed.")

        if any(w < 0.0 for w in user_weights):
            logger.info(
                "[AnimaCrossAttn] negative artist weights detected; those "
                "artists subtract their style direction (style subtraction)."
            )
            if combine_mode == COMBINE_CONCAT:
                logger.warning(
                    "[AnimaCrossAttn] combine=concat treats negative weights as "
                    "sign-flipped K/V tokens, not style subtraction. Delta-space "
                    "style subtraction only exists in output_avg/lowrank_avg."
                )

        if fusion_mode == FUSION_BASE_PRESERVE and float(strength) < 0.3:
            logger.info(
                "[AnimaCrossAttn] fusion=base_preserve at strength=%.2f (<0.3) "
                "is very subtle; consider strength >= 0.7.",
                float(strength),
            )

        if float(strength) > 1.0:
            logger.info(
                "[AnimaCrossAttn] strength=%.2f > 1.0 enters extrapolation: "
                "out = base + %.2f * (artist - base). %s.",
                float(strength),
                float(strength),
                "Recommended range 1.5-2.5"
                if float(strength) <= 3.0
                else "Current value is high and may oversaturate",
            )

        if not normalize_w and n > 1 and combine_mode in (COMBINE_OUTPUT_AVG, COMBINE_LOWRANK_AVG):
            effective_weight_sum = sum(abs(w) for w in user_weights)
            if effective_weight_sum >= 4.0 and not has_explicit_weights:
                raise ValueError(
                    f"[AnimaCrossAttn] normalize_weights=False with {n} artists "
                    f"(effective weight sum {effective_weight_sum:.2f}) will "
                    f"visibly amplify the cross-attention output under "
                    f"combine={combine_mode} and almost always break the image.\n"
                    f"Pick one:\n"
                    f"  1) set normalize_weights=True in AnimaArtistOptions (recommended)\n"
                    f"  2) lower the linear strength via ::name::0.25 in the chain\n"
                    f"  3) switch combine_mode to concat (no weighted sum)"
                )
            elif effective_weight_sum > 1.5:
                logger.warning(
                    "[AnimaCrossAttn] normalize_weights=False and effective weight "
                    "sum %.2f (artists=%d, combine=%s); the output may be too "
                    "strong. Lower ::weight values, enable normalize, or use concat.",
                    effective_weight_sum,
                    n,
                    combine_mode,
                )

        try:
            dm = model.get_model_object("diffusion_model")
        except Exception:
            dm = model.model.diffusion_model

        ok, num_blocks, ctx_dim, msg = validate_model(dm)
        if not ok:
            raise ValueError(f"[AnimaCrossAttn] unsupported model: {msg}")
        if not hasattr(dm, "preprocess_text_embeds"):
            raise ValueError("[AnimaCrossAttn] this is not an Anima model (missing preprocess_text_embeds)")

        artist_layer_routes, has_artist_layer_routes = resolve_artist_layer_routes(
            layer_route_texts,
            num_blocks,
        )
        if len(artist_layer_routes) < n:
            artist_layer_routes.extend([None] * (n - len(artist_layer_routes)))
        elif len(artist_layer_routes) > n:
            artist_layer_routes = artist_layer_routes[:n]

        artist_timing_percent_routes, has_artist_timing_routes = resolve_artist_timing_routes(
            timing_route_texts,
        )
        if len(artist_timing_percent_routes) < n:
            artist_timing_percent_routes.extend([None] * (n - len(artist_timing_percent_routes)))
        elif len(artist_timing_percent_routes) > n:
            artist_timing_percent_routes = artist_timing_percent_routes[:n]

        target_blocks = resolve_target_blocks_from_options(adv, num_blocks, strict=True)
        external_patches = describe_external_cross_attn_patches(dm, target_blocks)
        if external_patches and not adv.get("compatibility_mode", False):
            logger.warning(
                "[AnimaCrossAttn] possible external cross-attn wrappers detected: %s. "
                "If the artist effect weakens or disappears, try the "
                "compatibility_safe preset.",
                "; ".join(external_patches[:8]),
            )

        sigma_range = None
        if use_sigma_range:
            try:
                ms = model.get_model_object("model_sampling")
                s_at_start = float(ms.percent_to_sigma(start_percent))
                s_at_end = float(ms.percent_to_sigma(end_percent))
                lo, hi = sorted([s_at_end, s_at_start])
                sigma_range = (lo, hi)
            except Exception as e:
                logger.warning(
                    "[AnimaCrossAttn] failed to resolve the sigma range: %s. Step-range control is disabled.",
                    e,
                )
                sigma_range = None

        adv["stabilizer_min_sigma"] = None
        if use_stabilizer_window:
            try:
                ms = model.get_model_object("model_sampling")
                adv["stabilizer_min_sigma"] = float(ms.percent_to_sigma(adv["stabilizer_end_percent"]))
            except Exception as e:
                logger.warning(
                    "[AnimaCrossAttn] failed to resolve stabilizer_end_percent "
                    "%.3f: %s. Stabilizers stay active for the whole sampling pass.",
                    adv["stabilizer_end_percent"],
                    e,
                )
                adv["stabilizer_min_sigma"] = None

        artist_timing_routes = [None] * n
        if has_artist_timing_routes:
            try:
                ms = model.get_model_object("model_sampling")
                artist_timing_routes = []
                for timing in artist_timing_percent_routes:
                    if timing is None:
                        artist_timing_routes.append(None)
                        continue
                    artist_timing_routes.append(_percent_window_to_sigma_route(ms, timing))
            except Exception as e:
                logger.warning(
                    "[AnimaCrossAttn] failed to resolve per-artist timing sigma "
                    "ranges: %s. Timing routes are treated as always active.",
                    e,
                )
                artist_timing_routes = [None] * n
                has_artist_timing_routes = False

        need_sigma_capture = need_sigma_capture or has_artist_timing_routes

        m = model.clone()

        state = _build_runtime_state(
            enabled,
            fusion_mode,
            combine_mode,
            strength,
            apply_to_uncond,
            raws,
            ids_list,
            w_list,
            user_weights,
            labels,
            artist_layer_routes,
            has_artist_layer_routes,
            artist_timing_routes,
            has_artist_timing_routes,
            normalize_w,
            has_explicit_weights,
            preset_name,
            adv,
            dm,
            sigma_range,
            external_patches,
        )

        # The sigma-capture wrapper is always installed: it is the single
        # run-start reset point every configuration relies on (disabled-layer
        # set, EMA/static caches, warning latches). The chaining warning stays
        # gated to sigma-dependent features so plain configs chain silently.
        prev = m.model_options.get("model_function_wrapper")
        if need_sigma_capture and prev is not None and not adv.get("compatibility_mode", False):
            logger.warning(
                "[AnimaCrossAttn] another model_function_wrapper is already "
                "installed; this node chains to it. If timing routes or "
                "stabilizers stop working, enable compatibility_safe or "
                "simplify the other wrapper nodes."
            )
        m.set_model_unet_function_wrapper(make_sigma_capture(state, prev))

        # If a previous Anima mixer node already patched some of these blocks
        # (dual-mixer chain), warn once: object patches replace, so the later
        # node wins on the overlap.
        existing_patches = getattr(m, "object_patches", None) or {}
        overlapped = [
            i for i in target_blocks if f"diffusion_model.blocks.{i}.cross_attn.forward" in existing_patches
        ]
        if overlapped:
            logger.warning(
                "[AnimaCrossAttn] another Anima Artist mixer node already patches "
                "blocks %d-%d on this model; the later node wins on overlapping "
                "blocks.",
                min(overlapped),
                max(overlapped),
            )

        for i in target_blocks:
            ca = dm.blocks[i].cross_attn
            current_forward = getattr(ca, "forward", None)

            # Check if already patched by this code (handles multiple sampler workflows)
            if hasattr(current_forward, "_anima_artist_mixer_forward_patch"):
                # Already patched - unwrap to get the true original
                inner = unwrap_cross_attn_forward(ca)
            else:
                # Not yet patched - proceed normally
                inner = unwrap_cross_attn_forward(unwrap_cross_attn(ca))

            wrapper = CrossAttnWrapper(
                inner,
                state,
                i,
                original_module=unwrap_cross_attn(ca),
            )
            m.add_object_patch(
                f"diffusion_model.blocks.{i}.cross_attn.forward",
                make_cross_attn_forward_patch(wrapper),
            )

        return (m, base_cond_out)


class AnimaArtistPresetApply:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "artist_pack": ("ANIMA_PACK",),
                "preset": (
                    "ANIMA_PRESET",
                    {
                        "tooltip": (
                            "Preset payload from Anima Artist Preset, Starter, or "
                            "Recipe Load. It owns combine/fusion/strength."
                        ),
                    },
                ),
                "enabled": ("BOOLEAN", {"default": True}),
                "apply_to_uncond": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Default False. Applying style to uncond usually breaks CFG.",
                    },
                ),
            },
            "optional": {
                "advanced_options": (
                    "ANIMA_OPTS",
                    {
                        "tooltip": (
                            "Optional explicit option override. Leave disconnected "
                            "when the preset alone is enough."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING")
    RETURN_NAMES = ("model", "base_prompt")
    FUNCTION = "apply"
    CATEGORY = "Anima/Setup"

    def apply(self, model, artist_pack, preset, enabled, apply_to_uncond, advanced_options=None):
        return AnimaArtistCrossAttn().patch(
            model,
            artist_pack,
            COMBINE_OUTPUT_AVG,
            FUSION_INTERPOLATE,
            1.0,
            enabled,
            apply_to_uncond,
            advanced_options=advanced_options,
            preset=preset,
        )


# Backward-compatible re-exports (v27.4 module split). Safe against the
# import cycle: anima_mixer/__init__ imports this module before the
# split-off modules, and every name above is defined before these run.
from .nodes_pack import AnimaArtistBasic  # noqa: E402,F401
from .nodes_pack import AnimaArtistPack  # noqa: E402,F401
from .nodes_pack import _BASIC_PACK_CACHE  # noqa: E402,F401
from .nodes_pack import _BASIC_PACK_CACHE_LIMIT  # noqa: E402,F401
from .nodes_probe import PROBE_REGISTRY  # noqa: E402,F401
from .nodes_probe import _PROBE_REGISTRY_LIMIT  # noqa: E402,F401
from .nodes_probe import AnimaArtistProbe  # noqa: E402,F401
from .nodes_probe import AnimaArtistProbeReport  # noqa: E402,F401
from .nodes_probe import _ProbeCrossAttnWrapper  # noqa: E402,F401
from .nodes_probe import _registry_store  # noqa: E402,F401
from .nodes_probe import _suggest_layer_range  # noqa: E402,F401
