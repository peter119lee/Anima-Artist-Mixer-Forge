"""Options nodes: the simple and expert runtime-option builders."""

from .constants import (
    LAYER_MODE_AUTO,
    LAYER_MODE_CHOICES,
    ANCHOR_LAYER_THRESHOLD_DISABLED,
    ANCHOR_SEEDS_MAX,
    CONTRIB_BALANCE_ALPHA_DEFAULT,
    MIXED_DELTA_CAP_RATIO_DEFAULT,
    MIXED_DELTA_CAP_RATIO_MAX,
    NORM_LOCK_ROW,
    NORM_LOCK_SCOPE_BOTH,
    NORM_LOCK_SCOPE_MIXED,
    NORM_LOCK_SCOPE_PER_ARTIST,
    NORM_LOCK_TOKEN,
    MAX_ARTISTS,
    STATIC_CAPTURE_BLEND_ALPHA_DEFAULT,
    STATIC_CAPTURE_K_DEFAULT,
    STATIC_CAPTURE_K_MAX,
    STATIC_CAPTURE_MODE_CHOICES,
    STATIC_CAPTURE_MODE_OUTPUT,
)
from .options import (
    layer_filter_for_mode,
)


class AnimaArtistSimpleOptions:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "normalize_weights": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "Recommended on. If artist_chain uses ::weight syntax, "
                            "explicit weights stay absolute at runtime."
                        ),
                    },
                ),
                "layer_mode": (
                    LAYER_MODE_CHOICES,
                    {
                        "default": LAYER_MODE_AUTO,
                        "tooltip": "Global layer shortcut. Keep auto/all unless you need a narrower range.",
                    },
                ),
                "start_percent": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.001,
                        "tooltip": "Sampling-progress start. 0.0 = beginning.",
                    },
                ),
                "end_percent": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.001,
                        "tooltip": "Sampling-progress end. 1.0 = end.",
                    },
                ),
                "custom_layer_filter": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "Used only when layer_mode=custom. Example: 0,3,5-10,-1",
                    },
                ),
                "compatibility_mode": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "Enable only when regional prompts or another attention "
                            "patcher conflicts with the mixer."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("ANIMA_OPTS",)
    RETURN_NAMES = ("advanced_options",)
    FUNCTION = "build"
    CATEGORY = "Anima/Setup"

    def build(
        self,
        normalize_weights,
        layer_mode,
        start_percent,
        end_percent,
        custom_layer_filter="",
        compatibility_mode=False,
    ):
        return (
            {
                "normalize_weights": bool(normalize_weights),
                "layer_filter": layer_filter_for_mode(layer_mode, custom_layer_filter),
                "start_percent": float(start_percent),
                "end_percent": float(end_percent),
                "compatibility_mode": bool(compatibility_mode),
            },
        )


class AnimaArtistOptions:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "start_block": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 63,
                        "step": 1,
                        "tooltip": "First patched block (inclusive). 0 = first layer.",
                    },
                ),
                "end_block": (
                    "INT",
                    {
                        "default": -1,
                        "min": -1,
                        "max": 63,
                        "step": 1,
                        "tooltip": "Last patched block (inclusive). -1 = last layer.",
                    },
                ),
                "start_percent": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.001,
                        "tooltip": "Sampling-progress start. 0.0 = beginning.",
                    },
                ),
                "end_percent": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.001,
                        "tooltip": "Sampling-progress end. 1.0 = end of sampling.",
                    },
                ),
                "normalize_weights": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "True: weights become relative proportions. False: weights "
                            "act as independent strengths.\n\n"
                            "If the artist_chain uses ::weight syntax, this switch is "
                            "bypassed at runtime (explicit weights stay absolute)."
                        ),
                    },
                ),
                "artist_ema_alpha": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 0.95,
                        "step": 0.05,
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
                    },
                ),
                "lowrank_k": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": MAX_ARTISTS,
                        "step": 1,
                        "tooltip": (
                            "LoRA-style low-rank dimension (combine_mode=lowrank_avg only).\n"
                            "Projects N artist deltas onto the top-k principal directions.\n"
                            "k=1: single consensus direction, most stable, most homogeneous\n"
                            "k=2-3: keeps main directions, allows some per-artist variety\n"
                            "k>=N: equivalent to output_avg (no projection)\n"
                            "Falls back to output_avg automatically when N=1."
                        ),
                    },
                ),
                "artist_static_capture": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "H' temporal average: accumulate artist attention over the\n"
                            "first K steps, then freeze and reuse the average (also a\n"
                            "30-50% speedup). K set by static_capture_k.\n"
                            "Compatible: output_avg / lowrank_avg + interpolate / base_preserve.\n"
                            "Incompatible: fusion=concat_with_base (ignored automatically).\n"
                            "Mutually exclusive with EMA (EMA is ignored when on)."
                        ),
                    },
                ),
                "static_capture_k": (
                    "INT",
                    {
                        "default": STATIC_CAPTURE_K_DEFAULT,
                        "min": 1,
                        "max": STATIC_CAPTURE_K_MAX,
                        "step": 1,
                        "tooltip": (
                            "Steps accumulated before freezing (artist_static_capture only).\n"
                            "K=1: single-point cache (fastest, most cross-seed drift)\n"
                            "K=6: recommended default\n"
                            "K=8-12: stronger drift suppression, more warmup cost\n"
                            "If total steps < K it keeps averaging without freezing."
                        ),
                    },
                ),
                "static_capture_mode": (
                    STATIC_CAPTURE_MODE_CHOICES,
                    {
                        "default": STATIC_CAPTURE_MODE_OUTPUT,
                        "tooltip": (
                            "What static_capture freezes.\n"
                            "output: freeze the full artist attention output (legacy, strongest lock)\n"
                            "delta: freeze artist-base delta and add it to the current base output "
                            "(less content smearing, weaker lock)\n"
                            "blend: interpolate output and delta paths using static_capture_blend_alpha\n"
                            "blend_perp: like blend, but only reintroduces base motion perpendicular "
                            "to the frozen style delta. Advanced A/B mode; not the stable_seed default."
                        ),
                    },
                ),
                "static_capture_blend_alpha": (
                    "FLOAT",
                    {
                        "default": STATIC_CAPTURE_BLEND_ALPHA_DEFAULT,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": (
                            "Used by static_capture mode blend / blend_perp.\n"
                            "0 = legacy output freeze, 1 = maximum base-motion return.\n"
                            "Lower values keep the style lock stronger; higher values protect "
                            "content motion but can reintroduce drift."
                        ),
                    },
                ),
                "artist_anchor_q": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "Anchor-Q: artist cross-attention uses a fixed-seed anchor\n"
                            "hidden state as Q, decoupling style mixing from the user seed.\n"
                            "The strongest built-in stabilizer, but not a full seed lock.\n\n"
                            "Cost: one extra full forward on the first generation; same\n"
                            "prompt across seeds then hits the cache for free.\n\n"
                            "Mutually exclusive with static_capture (anchor wins).\n"
                            "Incompatible with fusion=concat_with_base.\n\n"
                            "Risk: stroke placement may track the current image less\n"
                            "closely; raise anchor_user_blend if that bothers you."
                        ),
                    },
                ),
                "anchor_seeds_count": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": ANCHOR_SEEDS_MAX,
                        "step": 1,
                        "tooltip": (
                            "Number of fixed seeds for the anchor pre-run (anchor_q only).\n"
                            "1: single seed\n"
                            "2-4: average several anchors to reduce single-seed bias.\n"
                            "Pre-run time scales with the count; cached afterwards."
                        ),
                    },
                ),
                "anchor_user_blend": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": (
                            "Anchor / user-x blend (anchor_q only).\n"
                            "Q = blend * user_x + (1-blend) * anchor_x\n"
                            "0.0: pure anchor (most stable across seeds)\n"
                            "0.3-0.5: balanced stability vs stroke fit\n"
                            "1.0: pure user x (equivalent to anchor_q off)"
                        ),
                    },
                ),
                "anchor_deep_layer_threshold": (
                    "INT",
                    {
                        "default": ANCHOR_LAYER_THRESHOLD_DISABLED,
                        "min": ANCHOR_LAYER_THRESHOLD_DISABLED,
                        "max": 64,
                        "step": 1,
                        "tooltip": (
                            "Use the anchor only in shallow layers (anchor_q only).\n"
                            "-1: every layer uses the anchor\n"
                            "N>=0: layers < N use the anchor, layers >= N use user x.\n"
                            "Shallow layers set style direction; deep layers fit strokes.\n"
                            "Example: 28-block model with N=14 anchors the first half."
                        ),
                    },
                ),
                "anchor_refresh_each_step": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "Refresh the fixed-seed anchor at every sampling step "
                            "instead of only the first step.\n"
                            "This gives the anchor a timestep-matched Q reference "
                            "and can reduce drift further, but costs extra forwards "
                            "every step. Advanced A/B option."
                        ),
                    },
                ),
                "stabilizer_end_percent": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": (
                            "Sampling progress where cache-based stabilizers stop. "
                            "Applies to EMA, static_capture, and anchor_q.\n"
                            "1.0: stabilizers run for the whole sampling pass.\n"
                            "0.4-0.6: useful when late-step samplers need dynamic "
                            "step-to-step motion."
                        ),
                    },
                ),
            },
            "optional": {
                "layer_filter": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": (
                            "Advanced layer selection. Comma-separated block indices, "
                            "ranges and negative indices supported.\n"
                            "Example: '0,3,5-10,-1'. Overrides start_block/end_block. "
                            "Empty = inactive."
                        ),
                    },
                ),
                "compatibility_mode": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "Compatibility-safe mode. Forces concat + concat_with_base "
                            "and disables EMA / static_capture / anchor_q to minimize "
                            "conflicts with regional prompting, Forge Couple-style "
                            "routing, and other attention patch nodes."
                        ),
                    },
                ),
                "max_batch_artists": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": MAX_ARTISTS,
                        "step": 1,
                        "tooltip": (
                            "Cap on how many artists run in one batched forward.\n"
                            "0 = automatic: on GPU the chunk size adapts to free "
                            "VRAM (v27.5); elsewhere no cap.\n"
                            "Set 2-8 to force a fixed cap with many artists at "
                            "high resolution."
                        ),
                    },
                ),
                "artist_q_reuse": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "Experimental speed-up: project attention Q once per "
                            "step and reuse it for every artist K/V (numerically "
                            "validated on first use).\n"
                            "OFF by default: the fp16 kernel difference shifts "
                            "same-seed renders (~17% of pixels in live A/B) and "
                            "it bypasses TeaCache-style attention patches."
                        ),
                    },
                ),
                "low_vram_cache": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "Store static-capture and anchor caches in system RAM "
                            "instead of VRAM. Saves hundreds of MB at high resolution "
                            "for a small per-step transfer cost."
                        ),
                    },
                ),
                "match_base_norm": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "Rescale the mixed artist attention output to the base "
                            "output's RMS energy (clamped to 0.5-2.0x).\n"
                            "Keeps the style direction but stops activation-energy "
                            "mismatch from compounding across layers. Enable this "
                            "explicitly when you want v26 norm-lock stabilization."
                        ),
                    },
                ),
                "anchor_base_norm_ref": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "When anchor_q and match_base_norm are both enabled, "
                            "match artist RMS against the fixed-seed anchor base "
                            "output instead of the current seed's base output.\n"
                            "This reduces cross-seed style-strength drift more than "
                            "standard match_base_norm, but may make strokes follow "
                            "the anchor reference more."
                        ),
                    },
                ),
                "norm_lock_mode": (
                    [NORM_LOCK_TOKEN, NORM_LOCK_ROW],
                    {
                        "default": NORM_LOCK_TOKEN,
                        "tooltip": (
                            "Granularity for match_base_norm.\n"
                            "token: match each image token's RMS to base (strongest "
                            "local style-strength stability)\n"
                            "row: legacy whole-row RMS matching"
                        ),
                    },
                ),
                "norm_lock_scope": (
                    [
                        NORM_LOCK_SCOPE_PER_ARTIST,
                        NORM_LOCK_SCOPE_MIXED,
                        NORM_LOCK_SCOPE_BOTH,
                    ],
                    {
                        "default": NORM_LOCK_SCOPE_PER_ARTIST,
                        "tooltip": (
                            "Where to apply norm locking.\n"
                            "per_artist: normalize each artist output before mixing, "
                            "so one seed-specific artist spike cannot dominate\n"
                            "mixed: normalize only the final mixed output (legacy)\n"
                            "both: strongest clamp, highest chance of over-uniform style"
                        ),
                    },
                ),
                "contribution_balance": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "Balance each artist's measured delta against the base "
                            "before mixing. Optional guard for seed-specific artist "
                            "dominance flips."
                        ),
                    },
                ),
                "contribution_balance_alpha": (
                    "FLOAT",
                    {
                        "default": CONTRIB_BALANCE_ALPHA_DEFAULT,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": (
                            "Strength of contribution_balance.\n0 = off, 1 = full per-token delta balancing."
                        ),
                    },
                ),
                "mixed_delta_cap": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "Inference-time guard for style drift. Limits the final "
                            "mixed artist delta relative to the base attention energy "
                            "before fusion. Default off for A/B testing."
                        ),
                    },
                ),
                "mixed_delta_cap_ratio": (
                    "FLOAT",
                    {
                        "default": MIXED_DELTA_CAP_RATIO_DEFAULT,
                        "min": 0.0,
                        "max": MIXED_DELTA_CAP_RATIO_MAX,
                        "step": 0.05,
                        "tooltip": (
                            "Maximum final artist-delta RMS as a multiple of base RMS.\n"
                            "Lower values preserve composition more; higher values "
                            "allow stronger style changes."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("ANIMA_OPTS",)
    RETURN_NAMES = ("advanced_options",)
    FUNCTION = "build"
    CATEGORY = "Anima/Setup"

    def build(
        self,
        start_block,
        end_block,
        start_percent,
        end_percent,
        normalize_weights,
        artist_ema_alpha=0.0,
        lowrank_k=1,
        artist_static_capture=False,
        static_capture_k=STATIC_CAPTURE_K_DEFAULT,
        static_capture_mode=STATIC_CAPTURE_MODE_OUTPUT,
        static_capture_blend_alpha=STATIC_CAPTURE_BLEND_ALPHA_DEFAULT,
        artist_anchor_q=False,
        anchor_seeds_count=1,
        anchor_user_blend=0.0,
        anchor_deep_layer_threshold=ANCHOR_LAYER_THRESHOLD_DISABLED,
        anchor_refresh_each_step=False,
        stabilizer_end_percent=1.0,
        layer_filter="",
        compatibility_mode=False,
        max_batch_artists=0,
        artist_q_reuse=False,
        low_vram_cache=False,
        match_base_norm=False,
        anchor_base_norm_ref=False,
        norm_lock_mode=NORM_LOCK_TOKEN,
        norm_lock_scope=NORM_LOCK_SCOPE_PER_ARTIST,
        contribution_balance=False,
        contribution_balance_alpha=CONTRIB_BALANCE_ALPHA_DEFAULT,
        mixed_delta_cap=False,
        mixed_delta_cap_ratio=MIXED_DELTA_CAP_RATIO_DEFAULT,
    ):
        return (
            {
                "start_block": int(start_block),
                "end_block": int(end_block),
                "start_percent": float(start_percent),
                "end_percent": float(end_percent),
                "normalize_weights": bool(normalize_weights),
                "artist_ema_alpha": float(artist_ema_alpha),
                "lowrank_k": int(lowrank_k),
                "artist_static_capture": bool(artist_static_capture),
                "static_capture_k": int(static_capture_k),
                "static_capture_mode": str(static_capture_mode or STATIC_CAPTURE_MODE_OUTPUT),
                "static_capture_blend_alpha": float(static_capture_blend_alpha),
                "artist_anchor_q": bool(artist_anchor_q),
                "anchor_seeds_count": int(anchor_seeds_count),
                "anchor_user_blend": float(anchor_user_blend),
                "anchor_deep_layer_threshold": int(anchor_deep_layer_threshold),
                "anchor_refresh_each_step": bool(anchor_refresh_each_step),
                "stabilizer_end_percent": float(stabilizer_end_percent),
                "layer_filter": str(layer_filter or ""),
                "compatibility_mode": bool(compatibility_mode),
                "max_batch_artists": int(max_batch_artists),
            "artist_q_reuse": bool(artist_q_reuse),
                "low_vram_cache": bool(low_vram_cache),
                "match_base_norm": bool(match_base_norm),
                "anchor_base_norm_ref": bool(anchor_base_norm_ref),
                "norm_lock_mode": str(norm_lock_mode or NORM_LOCK_TOKEN),
                "norm_lock_scope": str(norm_lock_scope or NORM_LOCK_SCOPE_PER_ARTIST),
                "contribution_balance": bool(contribution_balance),
                "contribution_balance_alpha": float(contribution_balance_alpha),
                "mixed_delta_cap": bool(mixed_delta_cap),
                "mixed_delta_cap_ratio": float(mixed_delta_cap_ratio),
            },
        )


class AnimaArtistStyleBalance:
    """Compat shim for upstream An1X3R/Anima-Artist-Mixer 26.x (MIT).

    Upstream ships a one-dial ``style_balance`` node; the forge's
    ``contribution_balance`` controller covers the same goal (artist
    dominance equalization) with row-masked, weight-aware math. This node
    keeps upstream workflows loadable by mapping the dial onto that
    controller. At 0.0 it is a pure passthrough so it never overrides an
    Options node earlier in the chain.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "style_balance": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": (
                            "Reduces seed-to-seed artist dominance drift by evening "
                            "artist delta strength before user weights apply.\n"
                            "Maps onto contribution_balance(_alpha); 0.0 = off "
                            "(passthrough)."
                        ),
                    },
                ),
            },
            "optional": {
                "advanced_options": ("ANIMA_OPTS",),
            },
        }

    RETURN_TYPES = ("ANIMA_OPTS",)
    RETURN_NAMES = ("advanced_options",)
    FUNCTION = "build"
    CATEGORY = "Anima/Setup"

    def build(self, style_balance=0.0, advanced_options=None):
        opts = dict(advanced_options or {})
        balance = max(0.0, min(1.0, float(style_balance)))
        if balance > 0.0:
            opts["contribution_balance"] = True
            opts["contribution_balance_alpha"] = balance
        return (opts,)
