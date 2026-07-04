"""Cross-attention stabilizer internals: EMA, static capture, norm lock,
contribution balance, delta cap, anchor-Q, and their cache helpers.
"""


import logging

import torch

from .constants import (
    CONTRIB_BALANCE_MAX_SCALE,
    CONTRIB_BALANCE_MIN_SCALE,
    FUSION_BASE_PRESERVE,
    FUSION_CONCAT_WITH_BASE,
    FUSION_INTERPOLATE,
    NORM_LOCK_ROW,
    NORM_LOCK_SCOPE_BOTH,
    NORM_LOCK_SCOPE_MIXED,
    NORM_LOCK_SCOPE_PER_ARTIST,
    NORM_LOCK_TOKEN,
    MIXED_DELTA_CAP_RATIO_DEFAULT,
    STATIC_CAPTURE_BLEND_ALPHA_DEFAULT,
    STATIC_CAPTURE_K_DEFAULT,
    STATIC_CAPTURE_MODE_BLEND,
    STATIC_CAPTURE_MODE_BLEND_PERP,
    STATIC_CAPTURE_MODE_DELTA,
    STATIC_CAPTURE_MODE_OUTPUT,
    ANCHOR_LAYER_THRESHOLD_DISABLED,
)
from .math_utils import (
    project_perpendicular,
)
from .patching import (
    _in_stabilizer_window,
)

logger = logging.getLogger(__name__)


def _cache_store(tensor, low_vram):
    """Detach a tensor for caching, optionally offloading to CPU."""
    t = tensor.detach()
    return t.cpu() if low_vram else t


def _cache_load(tensor, like):
    """Bring a cached tensor back to the compute device/dtype of ``like``."""
    return tensor.to(device=like.device, dtype=like.dtype)


def _resolve_norm_lock_mode(mode):
    mode = str(mode or NORM_LOCK_TOKEN).strip().lower()
    if mode == NORM_LOCK_ROW:
        return NORM_LOCK_ROW
    return NORM_LOCK_TOKEN


def _resolve_norm_lock_scope(scope):
    scope = str(scope or NORM_LOCK_SCOPE_PER_ARTIST).strip().lower()
    if scope in (NORM_LOCK_SCOPE_MIXED, NORM_LOCK_SCOPE_PER_ARTIST, NORM_LOCK_SCOPE_BOTH):
        return scope
    return NORM_LOCK_SCOPE_PER_ARTIST


def _resolve_static_capture_mode(mode):
    mode = str(mode or STATIC_CAPTURE_MODE_OUTPUT).strip().lower()
    if mode in (
        STATIC_CAPTURE_MODE_DELTA,
        STATIC_CAPTURE_MODE_BLEND,
        STATIC_CAPTURE_MODE_BLEND_PERP,
    ):
        return mode
    return STATIC_CAPTURE_MODE_OUTPUT


def _static_capture_blend_alpha(st):
    try:
        alpha = float(st.get("static_capture_blend_alpha", STATIC_CAPTURE_BLEND_ALPHA_DEFAULT))
    except (TypeError, ValueError):
        alpha = STATIC_CAPTURE_BLEND_ALPHA_DEFAULT
    return max(0.0, min(1.0, alpha))


def _static_capture_to_outputs(mode, cached, base_out, alpha):
    if mode == STATIC_CAPTURE_MODE_DELTA:
        return [base_out + delta for delta in cached]
    if mode == STATIC_CAPTURE_MODE_BLEND:
        return [
            (1.0 - alpha) * pair[0] + alpha * (base_out + pair[1])
            for pair in cached
        ]
    if mode == STATIC_CAPTURE_MODE_BLEND_PERP:
        outs = []
        for pair in cached:
            frozen_out = pair[0]
            frozen_delta = pair[1]
            base_motion = base_out + frozen_delta - frozen_out
            motion_perp = project_perpendicular(base_motion, frozen_delta)
            outs.append(frozen_out + alpha * motion_perp)
        return outs
    return cached


def _static_capture_values_to_cache(mode, outs, base_out):
    if mode == STATIC_CAPTURE_MODE_DELTA:
        base_f32 = base_out.to(torch.float32)
        return [out.to(torch.float32) - base_f32 for out in outs]
    if mode in (STATIC_CAPTURE_MODE_BLEND, STATIC_CAPTURE_MODE_BLEND_PERP):
        base_f32 = base_out.to(torch.float32)
        return [
            torch.stack([out.to(torch.float32), out.to(torch.float32) - base_f32], dim=0)
            for out in outs
        ]
    return outs


def _row_mask_like(mask, ref):
    """Boolean per-row mask shaped to broadcast against ``ref`` (B, 1, 1, ...)."""
    return torch.tensor(mask, device=ref.device, dtype=torch.bool).view(
        len(mask), *([1] * (ref.dim() - 1))
    )


class StabilizerMixin:
    """Stabilizer methods shared into CrossAttnWrapper (state in self._st)."""

    # ------------------------------------------------------------------ EMA

    def _maybe_reset_ema(self):
        """A sigma jump upward means a new sampling run -> reset EMA cache."""
        st = self._st
        cur = st.get("current_sigma")
        if cur is None:
            return
        prev = st.get("_ema_last_sigma")
        if prev is None or cur > prev + 1e-3:
            st["_ema_cache"] = {}
        st["_ema_last_sigma"] = cur

    def _apply_ema(self, artist_total, fusion_mode, fp=None):
        """Cross-step EMA smoothing (fusion in {interpolate, base_preserve}).

        concat_with_base never produces an artist_total, and static capture
        already freezes artist outputs, so EMA is skipped in both cases. The
        cache is keyed by (layer, forward fingerprint) so several forwards at
        one sigma (multi-cond / VRAM-split batches) do not blend together, and
        entries honor low_vram_cache offloading.
        """
        st = self._st
        if st.get("artist_static_capture", False):
            return artist_total
        ema_alpha = float(st.get("artist_ema_alpha", 0.0))
        ema_compatible = fusion_mode in (FUSION_INTERPOLATE, FUSION_BASE_PRESERVE)
        if ema_alpha <= 0.0 or not ema_compatible:
            return artist_total
        if st.get("current_sigma") is None:
            self._warn_no_sigma()
            return artist_total
        if not _in_stabilizer_window(st):
            return artist_total
        self._maybe_reset_ema()
        low_vram = bool(st.get("low_vram_cache", False))
        cache = st.setdefault("_ema_cache", {})
        key = (self._idx, fp)
        prev = cache.get(key)
        if prev is not None and prev.shape == artist_total.shape:
            prev = _cache_load(prev, artist_total)
            artist_total = ema_alpha * prev + (1.0 - ema_alpha) * artist_total
        cache[key] = _cache_store(artist_total, low_vram)
        return artist_total

    # -------------------------------------------------------- static capture

    def _maybe_reset_static(self):
        """Reset the static cache when a new sampling run starts.

        Tracks the last sigma seen every call (like the EMA reset). Within a
        run sigma only decreases, so this never fires; a re-queue with the same
        schedule jumps sigma back up on the first step and clears the frozen
        artist outputs, so they cannot leak across generations. The previous
        max-sigma tracking never reset when a re-queue repeated the schedule
        exactly (cur == prev_max).
        """
        st = self._st
        cur = st.get("current_sigma")
        if cur is None:
            return
        prev = st.get("_static_last_sigma")
        if prev is None or cur > prev + 1e-3:
            st["_static_cache"] = {}
        st["_static_last_sigma"] = cur

    def _get_artist_outputs_with_cache(self, x, context, rope_emb, t_opts,
                                       individuals, fusion_mode, base_out=None,
                                       extra_fp=None, fp=None):
        """H' temporal averaging: accumulate the first K steps, then freeze.

        Accumulation runs in fp32; returned tensors keep the model dtype. The
        static cache is keyed by (layer, forward fingerprint ``fp``) so several
        forwards at one sigma keep independent entries. The per-entry
        fingerprint (x.shape, n, mode, ``extra_fp``) invalidates on a
        resolution / artist-count / mode change; ``extra_fp`` also folds in the
        combined path's weight*fade so a mid-fade freeze cannot lock a stale
        weight. A sigma jump (new run) resets everything; repeated calls at the
        same sigma (CFG double forward) reuse the current average.
        """
        st = self._st
        low_vram = bool(st.get("low_vram_cache", False))
        if not st.get("artist_static_capture", False):
            return self._collect_artist_outputs(
                x, context, rope_emb, t_opts, individuals, fusion_mode
            )
        if st.get("current_sigma") is None:
            self._warn_no_sigma()
            return self._collect_artist_outputs(
                x, context, rope_emb, t_opts, individuals, fusion_mode
            )
        if not _in_stabilizer_window(st):
            return self._collect_artist_outputs(
                x, context, rope_emb, t_opts, individuals, fusion_mode
            )
        # static capture cannot work for concat_with_base (x changes every
        # step, and the artist attention includes the base context).
        if fusion_mode == FUSION_CONCAT_WITH_BASE:
            return self._collect_artist_outputs(
                x, context, rope_emb, t_opts, individuals, fusion_mode
            )
        mode = _resolve_static_capture_mode(st.get("static_capture_mode"))
        if mode in (
            STATIC_CAPTURE_MODE_DELTA,
            STATIC_CAPTURE_MODE_BLEND,
            STATIC_CAPTURE_MODE_BLEND_PERP,
        ) and base_out is None:
            mode = STATIC_CAPTURE_MODE_OUTPUT
        blend_alpha = _static_capture_blend_alpha(st)

        self._maybe_reset_static()
        cache = st.setdefault("_static_cache", {})
        n = len(individuals)
        entry_fp = (tuple(x.shape), n, mode, extra_fp)
        cache_key = (self._idx, fp)

        cur_sigma = st.get("current_sigma")
        sigma_key = round(float(cur_sigma), 4) if cur_sigma is not None else None

        entry = cache.get(cache_key)
        if entry is None or entry.get("_fp") != entry_fp:
            entry = {
                "_fp": entry_fp,
                "seen_sigmas": set(),
                "accumulator": None,
                "count": 0,
                "frozen": False,
                "frozen_outputs": None,
            }
            cache[cache_key] = entry

        if entry["frozen"]:
            frozen = [_cache_load(o, context) for o in entry["frozen_outputs"]]
            return _static_capture_to_outputs(mode, frozen, base_out, blend_alpha)

        # Same sigma re-entry (CFG second forward): return the current
        # average without recomputing or re-accumulating.
        if sigma_key is not None and sigma_key in entry["seen_sigmas"]:
            if entry["accumulator"] is not None and entry["count"] > 0:
                inv = 1.0 / entry["count"]
                averaged = [
                    _cache_load(a * inv, context) for a in entry["accumulator"]
                ]
                return _static_capture_to_outputs(mode, averaged, base_out, blend_alpha)
            return self._collect_artist_outputs(
                x, context, rope_emb, t_opts, individuals, fusion_mode
            )

        outs = self._collect_artist_outputs(
            x, context, rope_emb, t_opts, individuals, fusion_mode
        )
        to_cache = _static_capture_values_to_cache(mode, outs, base_out)
        if entry["accumulator"] is None:
            entry["accumulator"] = [
                _cache_store(o.to(torch.float32), low_vram) for o in to_cache
            ]
        else:
            for i, o in enumerate(to_cache):
                acc = entry["accumulator"][i]
                add = _cache_store(o.to(torch.float32), low_vram)
                entry["accumulator"][i] = acc + add
        entry["count"] += 1
        if sigma_key is not None:
            entry["seen_sigmas"].add(sigma_key)

        capture_k = int(self._st.get("static_capture_k", STATIC_CAPTURE_K_DEFAULT))
        if entry["count"] >= capture_k:
            inv = 1.0 / entry["count"]
            # Freeze in the model dtype: keeping fp32 here would double the
            # resident cache size for no accuracy benefit after averaging.
            entry["frozen_outputs"] = [
                _cache_store((a * inv).to(context.dtype), low_vram)
                for a in entry["accumulator"]
            ]
            entry["frozen"] = True
            entry["accumulator"] = None  # release memory
            entry["seen_sigmas"] = None
            frozen = [_cache_load(o, context) for o in entry["frozen_outputs"]]
            return _static_capture_to_outputs(mode, frozen, base_out, blend_alpha)

        inv = 1.0 / entry["count"]
        averaged = [_cache_load(a * inv, context) for a in entry["accumulator"]]
        return _static_capture_to_outputs(mode, averaged, base_out, blend_alpha)

    # ----------------------------------------------------------------- fusion

    def _apply_fusion(self, base_out, artist_total, mask, fusion_mode, strength):
        """Single fusion exit for the delta-space paths.

        Reached with fusion_mode in {interpolate, base_preserve}. The combine=
        concat single-forward path handles concat_with_base itself in
        _fwd_with_combined and never gets here; but combine=output_avg /
        lowrank_avg with fusion=concat_with_base DO reach this, where
        concat_with_base is not base_preserve and so falls through to the
        interpolate lerp below.
        """
        row_mask = _row_mask_like(mask, base_out)
        if fusion_mode == FUSION_BASE_PRESERVE:
            delta = artist_total - base_out
            delta_perp = project_perpendicular(delta, base_out)
            blended = base_out + strength * delta_perp
            return torch.where(row_mask, blended, base_out)

        blended = base_out * (1.0 - strength) + artist_total * strength
        return torch.where(row_mask, blended, base_out)

    def _match_base_norm(self, artist_total, base_out, mask, scale_floor=0.5):
        """Rescale the mixed artist output to the base output's RMS energy.

        The weighted artist mixture can carry noticeably different activation
        energy than the base output downstream blocks were trained on; the
        deviation compounds across layers and surfaces as seed-dependent
        style-strength swings. Token mode matches each image token's RMS;
        row mode preserves the legacy whole-row RMS behavior. Rows outside
        the injection mask keep scale 1.
        """
        norm_ref = self._get_anchor_base_norm_ref(base_out)
        mode = _resolve_norm_lock_mode(self._st.get("norm_lock_mode", NORM_LOCK_TOKEN))
        if mode == NORM_LOCK_ROW or artist_total.dim() < 3:
            dims = tuple(range(1, artist_total.dim()))
        else:
            dims = (-1,)
        base_rms = norm_ref.detach().to(torch.float32).pow(2).mean(
            dim=dims, keepdim=True).sqrt()
        artist_rms = artist_total.detach().to(torch.float32).pow(2).mean(
            dim=dims, keepdim=True).sqrt()
        scale = (base_rms / artist_rms.clamp(min=1e-6)).clamp(scale_floor, 2.0)
        row_mask = _row_mask_like(mask, scale)
        scale = torch.where(row_mask, scale, torch.ones_like(scale))
        return artist_total * scale.to(artist_total.dtype)

    def _get_anchor_base_norm_ref(self, base_out):
        st = self._st
        if not st.get("anchor_base_norm_ref", False):
            return base_out
        if st.get("_anchor_failed", False):
            return base_out
        cache = st.get("_anchor_base_cache", {})
        anchor_base = cache.get(self._idx)
        if anchor_base is None:
            return base_out
        if anchor_base.shape != base_out.shape:
            if anchor_base.shape[1:] == base_out.shape[1:]:
                ax_bsz = anchor_base.shape[0]
                bsz = base_out.shape[0]
                if bsz % ax_bsz == 0:
                    anchor_base = anchor_base.repeat(
                        bsz // ax_bsz, *([1] * (anchor_base.dim() - 1))
                    )
                elif ax_bsz % bsz == 0:
                    anchor_base = anchor_base[:bsz]
                else:
                    return base_out
            else:
                return base_out
        return anchor_base.to(device=base_out.device, dtype=base_out.dtype)

    def _contribution_balance_alpha(self):
        if not self._st.get("contribution_balance", False):
            return 0.0
        try:
            alpha = float(self._st.get("contribution_balance_alpha", 1.0))
        except (TypeError, ValueError):
            return 1.0
        return max(0.0, min(1.0, alpha))

    def _balance_artist_deltas(self, artist_outs, base_out, weights, mask):
        """Scale per-artist deltas toward a common target strength.

        Artist dominance flips happen when one seed makes a single artist's
        cross-attention delta much larger than the others. This controller
        measures each active artist's delta magnitude and nudges it toward
        the shared baseline implied by the group, so the user weights stay
        responsible for the final proportions instead of getting squared.
        """
        alpha = self._contribution_balance_alpha()
        if alpha <= 0.0 or len(artist_outs) < 2 or base_out is None:
            return artist_outs
        pos = [abs(float(w)) for w in weights]
        total_w = sum(pos)
        if total_w <= 1e-8:
            return artist_outs

        deltas = [(out - base_out).to(torch.float32) for out in artist_outs]
        dims = (-1,) if base_out.dim() >= 3 else tuple(range(1, base_out.dim()))
        strengths = [
            d.detach().pow(2).mean(dim=dims, keepdim=True).sqrt()
            for d in deltas
        ]
        active_strengths = []
        for s, w in zip(strengths, pos):
            if w <= 1e-8:
                continue
            active_strengths.append(s)
        if not active_strengths:
            return artist_outs
        total_strength = active_strengths[0].clone()
        for s in active_strengths[1:]:
            total_strength = total_strength + s
        target_strength = total_strength / float(len(active_strengths))

        row_mask = _row_mask_like(mask, base_out)
        balanced = []
        for out, delta, strength, w in zip(
            artist_outs, deltas, strengths, pos,
        ):
            if w <= 1e-8:
                balanced.append(out)
                continue
            scale = (target_strength / strength.clamp(min=1e-6)).clamp(
                CONTRIB_BALANCE_MIN_SCALE, CONTRIB_BALANCE_MAX_SCALE,
            )
            if alpha < 1.0:
                scale = 1.0 + alpha * (scale - 1.0)
            scale = torch.where(row_mask, scale, torch.ones_like(scale))
            adjusted = base_out + delta.to(out.dtype) * scale.to(out.dtype)
            balanced.append(adjusted)
        return balanced

    def _mixed_delta_cap_ratio(self):
        if not self._st.get("mixed_delta_cap", False):
            return 0.0
        try:
            ratio = float(self._st.get(
                "mixed_delta_cap_ratio", MIXED_DELTA_CAP_RATIO_DEFAULT,
            ))
        except (TypeError, ValueError):
            ratio = MIXED_DELTA_CAP_RATIO_DEFAULT
        return max(0.0, ratio)

    def _cap_mixed_delta(self, artist_total, base_out, mask, fusion_mode, strength):
        """Limit the effective mixed artist delta before final fusion."""
        ratio = self._mixed_delta_cap_ratio()
        if (
            ratio <= 0.0
            or base_out is None
            or fusion_mode not in (FUSION_INTERPOLATE, FUSION_BASE_PRESERVE)
        ):
            return artist_total

        delta = (artist_total - base_out).to(torch.float32)
        if fusion_mode == FUSION_BASE_PRESERVE:
            effective_delta = project_perpendicular(delta, base_out.to(torch.float32))
        else:
            effective_delta = delta

        final_strength = max(abs(float(strength)), 1e-6)
        dims = (-1,) if artist_total.dim() >= 3 else tuple(range(1, artist_total.dim()))
        base_rms = base_out.detach().to(torch.float32).pow(2).mean(
            dim=dims, keepdim=True,
        ).sqrt()
        delta_rms = effective_delta.detach().pow(2).mean(
            dim=dims, keepdim=True,
        ).sqrt()
        max_delta_rms = base_rms * float(ratio) / final_strength
        scale = torch.minimum(
            torch.ones_like(delta_rms),
            max_delta_rms / delta_rms.clamp(min=1e-6),
        )
        row_mask = _row_mask_like(mask, scale)
        scale = torch.where(row_mask, scale, torch.ones_like(scale))
        capped = base_out.to(torch.float32) + delta * scale
        return capped.to(artist_total.dtype)

    # ---------------------------------------------------------------- anchor

    def _get_anchor_q_x(self, x):
        """Return the Q source for artist attention (anchor-Q feature).

        Depending on configuration:
        - anchor disabled / pre-run failed / cache miss -> user x
        - layer >= anchor_deep_layer_threshold (when >= 0) -> user x
        - anchor_user_blend > 0 -> blend * x + (1-blend) * anchor_x
        - otherwise -> anchor_x

        Shape mismatches (e.g. batch changes) fall back to user x.
        """
        st = self._st
        if not st.get("artist_anchor_q", False):
            return x
        if st.get("_anchor_failed", False):
            return x
        if not _in_stabilizer_window(st):
            return x

        threshold = int(st.get("anchor_deep_layer_threshold", ANCHOR_LAYER_THRESHOLD_DISABLED))
        if threshold >= 0 and self._idx >= threshold:
            return x

        cache = st.get("_anchor_cache", {})
        anchor_x = cache.get(self._idx)
        if anchor_x is None:
            return x
        if anchor_x.shape != x.shape:
            if anchor_x.shape[1:] == x.shape[1:]:
                ax_bsz = anchor_x.shape[0]
                bsz = x.shape[0]
                if bsz % ax_bsz == 0:
                    anchor_x = anchor_x.repeat(bsz // ax_bsz, *([1] * (anchor_x.dim() - 1)))
                elif ax_bsz % bsz == 0:
                    anchor_x = anchor_x[:bsz]
                else:
                    return x
            else:
                return x
        anchor_x = anchor_x.to(device=x.device, dtype=x.dtype)

        blend = float(st.get("anchor_user_blend", 0.0))
        blend = max(0.0, min(1.0, blend))
        if blend > 0.0:
            return blend * x + (1.0 - blend) * anchor_x
        return anchor_x

