"""Pack nodes: artist splitting/encoding and the one-node Basic entry."""


import logging


from .chain_tools import lint_parsed_artists
from .constants import (
    COMBINE_OUTPUT_AVG,
    FUSION_INTERPOLATE,
    MAX_ARTISTS,
    PRESET_BALANCED,
    PRESET_RECOMMENDED_CHOICES,
)
from .options import build_preset_payload
from .parsing import (
    expand_prompt_weights,
    parse_artist_entries,
    parse_artist_layer_routes,
    parse_artist_timing_routes,
    split_artist_chain,
)

from .nodes_core import AnimaArtistCrossAttn

logger = logging.getLogger(__name__)


# Small FIFO memo so AnimaArtistBasic does not re-encode every artist when only
# preset/intensity change. Keyed by (chain, prompt); a hit additionally
# requires the pack's stored clip to be the SAME object (identity check) —
# id()-based keys are unsafe because CPython reuses a freed clip's address for
# its replacement, which would silently serve conditionings from the old
# text encoder.
_BASIC_PACK_CACHE = {}
_BASIC_PACK_CACHE_LIMIT = 4


class AnimaArtistPack:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "artist_chain": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": (
                        "Artist chain. Separate artists with commas or newlines.\n"
                        "Example: wlop, sakimichan, krenz\n"
                        "\n"
                        "Two weight syntaxes (they can coexist and stack):\n"
                        "  1) parentheses (wlop:1.5) - CLIP-side, non-linear\n"
                        "  2) ::weight ::wlop::1.5 - injection-side, linear\n"
                        "\n"
                        "Default weight 1.0; range [-4.0, 4.0]. Negative weights\n"
                        "subtract that artist's style (style subtraction).\n"
                        "::weight stacks with parentheses: ::(wlop:1.1)::0.8\n"
                        "Optional per-artist layer route: wlop@0-8, krenz@33%-67%\n"
                        "Optional per-artist timing: wlop@0-8%0.0-0.45\n"
                        "Use % for sampling timing; @0.0-0.5 is a layer range.\n"
                        "Optional timing fade: wlop%0.0-0.45~0.1 (smoothstep edges)\n"
                        "\n"
                        "When any artist uses ::weight, normalize_weights is\n"
                        "bypassed at runtime (explicit weights stay absolute)."
                    )
                }),
            },
            "optional": {
                "base_prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": (
                        "Main prompt (optional). Follows Anima's recommended layout: "
                        "artist first, then a newline, then the main prompt "
                        "('<artist>\\n<base_prompt>'). Leave empty to encode the "
                        "artist names alone."
                    ),
                }),
            },
        }

    RETURN_TYPES = ("ANIMA_PACK",)
    RETURN_NAMES = ("artist_pack",)
    FUNCTION = "pack"
    CATEGORY = "Anima/Setup"

    def pack(self, clip, artist_chain, base_prompt=""):
        raw_artist_chain = str(artist_chain or "").strip()
        parts = split_artist_chain(artist_chain)
        parts, timing_routes = parse_artist_timing_routes(parts)
        parts, layer_routes = parse_artist_layer_routes(parts)
        entries = parse_artist_entries(parts)
        names = [entry[0] for entry in entries]
        parsed_weights = [entry[1] for entry in entries]
        explicit_flags = [entry[2] for entry in entries]
        has_explicit = any(explicit_flags)

        for hint in lint_parsed_artists(names, layer_routes, timing_routes, raw_artist_chain):
            logger.warning("[AnimaArtistPack] %s", hint)

        # v26: Expand weight::target:: syntax in base_prompt
        base = expand_prompt_weights((base_prompt or "").strip())

        try:
            base_tokens = clip.tokenize(base)
            base_conditioning = clip.encode_from_tokens_scheduled(base_tokens)
        except Exception as e:
            raise ValueError(
                f"[AnimaArtistPack] failed to encode base_prompt (text={base!r}): {e}"
            )

        if not names:
            return ({
                "conditionings": [],
                "labels": [],
                "weights": [],
                "layer_routes": [],
                "timing_routes": [],
                "has_explicit_weights": False,
                "raw_artist_chain": raw_artist_chain,
                "base_prompt": base,
                "base_conditioning": base_conditioning,
                "clip": clip,
            },)

        if len(names) > MAX_ARTISTS:
            logger.warning(
                "[AnimaArtistPack] artist count %d exceeds the limit %d; truncating",
                len(names), MAX_ARTISTS,
            )
            names = names[:MAX_ARTISTS]
            parsed_weights = parsed_weights[:MAX_ARTISTS]
            explicit_flags = explicit_flags[:MAX_ARTISTS]
            layer_routes = layer_routes[:MAX_ARTISTS]
            timing_routes = timing_routes[:MAX_ARTISTS]
            # An explicit weight only on a truncated entry must not leak into
            # the surviving chain: it would disable normalization and skip the
            # weight-sum guard for 32 weight-1.0 artists.
            has_explicit = any(explicit_flags)

        conditionings = []
        for name in names:
            text = f"{name}\n{base}" if base else name
            try:
                tokens = clip.tokenize(text)
                cond = clip.encode_from_tokens_scheduled(tokens)
            except Exception as e:
                raise ValueError(
                    f"[AnimaArtistPack] encoding failed (text={text!r}): {e}"
                )
            conditionings.append(cond)

        if has_explicit:
            logger.info(
                "[AnimaArtistPack] %d artists carry ::weight syntax; the linear "
                "injection path will be used",
                sum(1 for flag in explicit_flags if flag),
            )

        return ({
            "conditionings": conditionings,
            "labels": names,
            "weights": parsed_weights,
            "layer_routes": layer_routes,
            "timing_routes": timing_routes,
            "has_explicit_weights": has_explicit,
            "raw_artist_chain": raw_artist_chain,
            "base_prompt": base,
            "base_conditioning": base_conditioning,
            "clip": clip,
        },)


class AnimaArtistBasic:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "artist_chain": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Artists separated by comma or newline.",
                }),
                "base_prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Main positive prompt. Do not repeat artist names here.",
                }),
                "preset": (PRESET_RECOMMENDED_CHOICES, {
                    "default": PRESET_BALANCED,
                    "tooltip": (
                        "Recommended modes only. Use Anima Artist Preset for "
                        "advanced modes.\n"
                        "balanced: original-compatible default\n"
                        "strong_style: stronger artist style\n"
                        "drift_auto: automatic low-drift route\n"
                        "prompt_passthrough: direct prompt/no mixer"
                    ),
                }),
                "intensity": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "Preset strength multiplier.",
                }),
                "enabled": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING")
    RETURN_NAMES = ("model", "base_prompt")
    FUNCTION = "apply"
    CATEGORY = "Anima/Basic"

    def apply(self, model, clip, artist_chain, base_prompt, preset, intensity, enabled):
        cache_key = (artist_chain, base_prompt)
        artist_pack = _BASIC_PACK_CACHE.get(cache_key)
        if artist_pack is None or artist_pack.get("clip") is not clip:
            artist_pack = AnimaArtistPack().pack(clip, artist_chain, base_prompt)[0]
            _BASIC_PACK_CACHE[cache_key] = artist_pack
            while len(_BASIC_PACK_CACHE) > _BASIC_PACK_CACHE_LIMIT:
                _BASIC_PACK_CACHE.pop(next(iter(_BASIC_PACK_CACHE)))
        payload = build_preset_payload(
            preset,
            intensity,
            normalize_weights=True,
            artist_count=len(artist_pack.get("labels") or []),
        )
        return AnimaArtistCrossAttn().patch(
            model,
            artist_pack,
            COMBINE_OUTPUT_AVG,
            FUSION_INTERPOLATE,
            1.0,
            enabled,
            False,
            preset=payload,
        )


