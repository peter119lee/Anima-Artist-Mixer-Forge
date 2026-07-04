"""Cross-attention wrapper: the runtime injection engine."""

import logging

import torch
import torch.nn as nn

from .constants import (
    COMBINE_LOWRANK_AVG,
    COMBINE_OUTPUT_AVG,
    FUSION_BASE_PRESERVE,
    FUSION_CONCAT_WITH_BASE,
    FUSION_INTERPOLATE,
    NORM_LOCK_SCOPE_BOTH,
    NORM_LOCK_SCOPE_MIXED,
    NORM_LOCK_SCOPE_PER_ARTIST,
    STATIC_CAPTURE_MODE_BLEND,
    STATIC_CAPTURE_MODE_BLEND_PERP,
    STATIC_CAPTURE_MODE_DELTA,
)
from .math_utils import (
    lowrank_rows_deterministic,
    timing_fade_factor,
)
from .parsing import normalize_weights
from .patching import (
    _forward_fingerprint,
    _in_stabilizer_window,  # noqa: F401  (shared-helper contract; used via StabilizerMixin)
    broadcast_batch,
    build_artists,
    in_sigma_range,
    resolve_mask,
)

from .wrapper_stabilizers import (  # noqa: E402
    StabilizerMixin,
    _cache_store,
    _resolve_norm_lock_scope,
    _resolve_static_capture_mode,
    _row_mask_like,
)
logger = logging.getLogger(__name__)


def _should_reraise(e):
    """Interrupts and OOM must propagate — never silently disable a layer.

    An out-of-memory error or a user interrupt is not an injection bug; the
    layer fallback would swallow it and mask the real problem.
    """
    for name in ("OutOfMemoryError",):
        cuda_oom = getattr(getattr(torch, "cuda", None), name, None)
        if cuda_oom is not None and isinstance(e, cuda_oom):
            return True
        torch_oom = getattr(torch, name, None)
        if torch_oom is not None and isinstance(e, torch_oom):
            return True
    try:
        from comfy.model_management import InterruptProcessingException
        if isinstance(e, InterruptProcessingException):
            return True
    except ImportError:
        pass
    return False


def _combine_concat(individuals, weights):
    parts = [a * float(w) for a, w in zip(individuals, weights)]
    return torch.cat(parts, dim=1)


class CrossAttnWrapper(StabilizerMixin, nn.Module):
    def __init__(self, original, shared_state, layer_idx):
        super().__init__()
        self.original = original
        self._st = shared_state
        self._idx = layer_idx

    def _warn_no_sigma(self):
        """One-shot warning when the stabilizers cannot see the sampling sigma.

        Another model wrapper may have replaced our sigma-capture hook; without
        the sigma we cannot tell run boundaries, so EMA/static capture would
        accumulate garbage. Skip them for the run instead.
        """
        st = self._st
        if not st.get("_warned_no_sigma", False):
            logger.warning(
                "[AnimaCrossAttn] cannot see the sampling sigma; EMA/static "
                "capture is disabled for this run (another model wrapper may "
                "have replaced the sigma-capture hook)."
            )
            st["_warned_no_sigma"] = True

    # ---------------------------------------------------------------- forward

    def forward(self, x, context=None, rope_emb=None, transformer_options=None):
        st = self._st
        transformer_options = transformer_options or {}

        # During the anchor pre-run: capture the layer input and run the
        # original cross-attention untouched.
        if st.get("_in_anchor_run", False):
            cache = st.setdefault("_anchor_cache", {})
            cache[self._idx] = _cache_store(
                x.clone(), bool(st.get("low_vram_cache", False))
            )
            base_out = self.original(x, context, rope_emb=rope_emb,
                                     transformer_options=transformer_options)
            if st.get("anchor_base_norm_ref", False):
                base_cache = st.setdefault("_anchor_base_cache", {})
                base_cache[self._idx] = _cache_store(
                    base_out.clone(), bool(st.get("low_vram_cache", False))
                )
            return base_out

        if not st.get("enabled", False) or context is None:
            return self.original(x, context, rope_emb=rope_emb,
                                 transformer_options=transformer_options)

        # A layer that failed is disabled for the rest of the run only. The set
        # lives in the shared run state (not on the wrapper) so the run-start
        # reset clears it — ComfyUI caches the patched model clone, so a
        # per-wrapper flag would stay stuck across queue runs.
        if self._idx in st.get("_disabled_layers", ()):
            return self.original(x, context, rope_emb=rope_emb,
                                 transformer_options=transformer_options)

        if not in_sigma_range(st):
            return self.original(x, context, rope_emb=rope_emb,
                                 transformer_options=transformer_options)

        try:
            return self._dispatch(x, context, rope_emb, transformer_options)
        except Exception as e:
            # Interrupts and OOM are not injection bugs; let them propagate.
            if _should_reraise(e):
                raise
            logger.exception(
                "[AnimaCrossAttn] L%d injection failed; this layer falls back "
                "to the original cross-attention: %s", self._idx, e,
            )
            st.setdefault("_disabled_layers", set()).add(self._idx)
            return self.original(x, context, rope_emb=rope_emb,
                                 transformer_options=transformer_options)

    def _dispatch(self, x, context, rope_emb, transformer_options):
        st = self._st
        individuals, _ = build_artists(st, context)
        combine_mode = st["combine_mode"]
        fusion_mode = st["fusion_mode"]
        strength = float(st["strength"])
        weights = st["user_weights"]
        fades = [1.0] * len(individuals)
        has_artist_routes = (
            st.get("has_artist_layer_routes", False)
            or st.get("has_artist_timing_routes", False)
        )
        if has_artist_routes:
            routes = st.get("artist_layer_routes") or []
            timings = st.get("artist_timing_routes") or []
            cur_sigma = st.get("current_sigma")
            filtered = []
            keep_zero_fades = combine_mode in (COMBINE_OUTPUT_AVG, COMBINE_LOWRANK_AVG)
            has_positive_fade = False
            for artist, weight, route, timing in zip(
                individuals, weights, routes, timings,
            ):
                if route is not None and self._idx not in route:
                    continue
                fade = timing_fade_factor(timing, cur_sigma)
                if fade <= 0.0:
                    if keep_zero_fades:
                        filtered.append((artist, weight, 0.0))
                    continue
                has_positive_fade = True
                filtered.append((artist, weight, fade))
            if not filtered or not has_positive_fade:
                return self.original(x, context, rope_emb=rope_emb,
                                     transformer_options=transformer_options)
            individuals = [item[0] for item in filtered]
            weights = [item[1] for item in filtered]
            fades = [item[2] for item in filtered]

        cou = transformer_options.get("cond_or_uncond") if isinstance(transformer_options, dict) else None
        bsz = context.shape[0]
        mask = resolve_mask(cou, bsz, st["apply_to_uncond"], st)

        if not any(mask):
            return self.original(x, context, rope_emb=rope_emb,
                                 transformer_options=transformer_options)

        # Per-forward fingerprint: several forwards can share one sigma (multi
        # positive conds, regional prompts, VRAM-split batches); keying the
        # EMA/static caches by it keeps them from cross-contaminating.
        fp = _forward_fingerprint(st, context)

        # lowrank_avg is meaningless for a single artist (no multi-artist
        # directions to project); it degrades to output_avg below.
        if combine_mode == COMBINE_LOWRANK_AVG and len(individuals) >= 2:
            return self._fwd_lowrank_avg(
                x, context, rope_emb, transformer_options,
                individuals, weights, fades, mask, fusion_mode, strength, fp=fp,
            )

        if combine_mode in (COMBINE_OUTPUT_AVG, COMBINE_LOWRANK_AVG):
            return self._fwd_output_avg(
                x, context, rope_emb, transformer_options,
                individuals, weights, fades, mask, fusion_mode, strength, fp=fp,
            )

        # concat never normalizes, so the fade multiplies the raw weight.
        combined = _combine_concat(
            individuals, [w * f for w, f in zip(weights, fades)],
        )
        # extra_fp folds the effective weight*fade into the static fingerprint
        # so a mid-fade freeze on the combined path cannot lock a stale weight.
        combined_fp = tuple(round(w * f, 6) for w, f in zip(weights, fades))
        return self._fwd_with_combined(
            x, context, rope_emb, transformer_options,
            combined, mask, fusion_mode, strength, fp=fp, extra_fp=combined_fp,
        )

    def _effective_weights(self, weights, fades):
        """Resolve normalization and timing fades into final weights.

        Normalization runs on the raw weights FIRST, then each artist's
        share is scaled by its fade factor — otherwise normalizing after the
        fade would cancel it whenever a layer has a single active artist
        (the common layer_scheduled case).

        Returns ``(ws, base_comp)`` where ``base_comp`` is the coefficient for
        the original base output in delta-space mixing:

            base + sum(w_i * (artist_i - base))
            == sum(w_i * artist_i) + (1 - sum(w_i)) * base

        For explicit ``::weight`` values this makes ``0.25`` a quarter style
        delta and ``-0.5`` a subtraction, not a raw output rescale.
        """
        should_normalize = self._st.get("normalize_weights", True)
        if should_normalize:
            ws_base = normalize_weights(weights)
        else:
            ws_base = list(weights)
        ws = [w * f for w, f in zip(ws_base, fades)]
        if should_normalize or self._st.get("has_explicit_weights", False):
            base_comp = 1.0 - sum(ws)
        else:
            base_comp = sum(w * (1.0 - f) for w, f in zip(ws_base, fades))
        return ws, base_comp

    def _fwd_output_avg(self, x, context, rope_emb, t_opts,
                        individuals, weights, fades, mask, fusion_mode, strength,
                        fp=None):
        bsz = context.shape[0]

        ws, base_comp = self._effective_weights(weights, fades)
        n = len(individuals)
        static_capture = self._st.get("artist_static_capture", False)
        norm_scope = _resolve_norm_lock_scope(self._st.get("norm_lock_scope", NORM_LOCK_SCOPE_PER_ARTIST))
        do_norm_lock = self._st.get("match_base_norm", False) and fusion_mode in (
            FUSION_INTERPOLATE, FUSION_BASE_PRESERVE
        )
        # The static-capture path must collect N independent outputs to cache
        # them. concat_with_base cannot be cached and skips this.
        force_collect = static_capture and fusion_mode != FUSION_CONCAT_WITH_BASE
        static_needs_base = (
            force_collect
            and _resolve_static_capture_mode(self._st.get("static_capture_mode"))
            in (
                STATIC_CAPTURE_MODE_DELTA,
                STATIC_CAPTURE_MODE_BLEND,
                STATIC_CAPTURE_MODE_BLEND_PERP,
            )
        )
        per_artist_lock = do_norm_lock and norm_scope in (
            NORM_LOCK_SCOPE_PER_ARTIST, NORM_LOCK_SCOPE_BOTH
        )
        mixed_lock = do_norm_lock and norm_scope in (
            NORM_LOCK_SCOPE_MIXED, NORM_LOCK_SCOPE_BOTH
        )
        balance_deltas = (
            self._contribution_balance_alpha() > 0.0
            and fusion_mode in (FUSION_INTERPOLATE, FUSION_BASE_PRESERVE)
            and n >= 2
        )
        cap_mixed_delta = (
            self._mixed_delta_cap_ratio() > 0.0
            and fusion_mode in (FUSION_INTERPOLATE, FUSION_BASE_PRESERVE)
        )

        skip_fusion = (
            fusion_mode == FUSION_INTERPOLATE and strength == 1.0 and all(mask)
        )
        base_out = None
        need_base_out = (
            do_norm_lock
            or balance_deltas
            or cap_mixed_delta
            or abs(base_comp) > 1e-6
            or fusion_mode == FUSION_BASE_PRESERVE
            or static_needs_base
            or not skip_fusion
        )
        if need_base_out:
            base_out = self.original(x, context, rope_emb=rope_emb, transformer_options=t_opts)

        artist_total = None
        if force_collect or per_artist_lock or balance_deltas:
            outs = self._get_artist_outputs_with_cache(
                x, context, rope_emb, t_opts, individuals, fusion_mode,
                base_out=base_out, fp=fp,
            )
            if per_artist_lock:
                outs = [
                    self._match_base_norm(out_i, base_out, mask, scale_floor=0.0)
                    for out_i in outs
                ]
            if balance_deltas:
                outs = self._balance_artist_deltas(outs, base_out, ws, mask)
                delta_total = None
                for out_i, w in zip(outs, ws):
                    delta_i = (out_i - base_out).to(torch.float32) * float(w)
                    delta_total = delta_i if delta_total is None else delta_total + delta_i
                artist_total = base_out + delta_total.to(base_out.dtype)
            else:
                for out_i, w in zip(outs, ws):
                    artist_total = out_i * w if artist_total is None else artist_total + out_i * w
        elif n >= 2 and not self._st.get("_disable_batched", False):
            try:
                q_x = self._get_anchor_q_x(x)
                artist_total = self._batched_artists_forward(
                    q_x, context, rope_emb, t_opts, individuals, ws, fusion_mode
                )
            except Exception as e:
                if not self._st.get("_warned_batched", False):
                    logger.warning(
                        "[AnimaCrossAttn] batched output_avg failed; "
                        "falling back to sequential mode: %s", e,
                    )
                    self._st["_warned_batched"] = True
                    self._st["_disable_batched"] = True
                artist_total = None
        if artist_total is None:
            q_x = self._get_anchor_q_x(x)
            for artist_i, w in zip(individuals, ws):
                artist_b = broadcast_batch(artist_i, bsz).to(
                    device=context.device, dtype=context.dtype)
                kv = torch.cat([context, artist_b], dim=1) \
                    if fusion_mode == FUSION_CONCAT_WITH_BASE else artist_b
                out_i = self.original(q_x, kv, rope_emb=rope_emb, transformer_options=t_opts)
                if per_artist_lock:
                    out_i = self._match_base_norm(out_i, base_out, mask, scale_floor=0.0)
                artist_total = out_i * w if artist_total is None else artist_total + out_i * w

        if abs(base_comp) > 1e-6 and not balance_deltas:
            # Return the unclaimed share to the base output. This covers both
            # timing fades and explicit weights smaller/larger than 1.0.
            artist_total = artist_total + base_comp * base_out

        artist_total = self._apply_ema(artist_total, fusion_mode, fp=fp)

        if mixed_lock:
            artist_total = self._match_base_norm(artist_total, base_out, mask)
        artist_total = self._cap_mixed_delta(
            artist_total, base_out, mask, fusion_mode, strength,
        )
        if skip_fusion:
            return artist_total
        return self._apply_fusion(base_out, artist_total, mask, fusion_mode, strength)

    # ------------------------------------------------------------- collectors

    def _collect_artist_outputs(self, x, context, rope_emb, t_opts,
                                individuals, fusion_mode):
        """Compute each artist's attention output. Returns list of (B, T, D)."""
        bsz = context.shape[0]
        n = len(individuals)
        q_x = self._get_anchor_q_x(x)
        if n >= 2 and not self._st.get("_disable_batched", False):
            try:
                return self._batched_artists_outputs_only(
                    q_x, context, rope_emb, t_opts, individuals, fusion_mode
                )
            except Exception as e:
                if not self._st.get("_warned_batched", False):
                    logger.warning(
                        "[AnimaCrossAttn] batched outputs failed; "
                        "falling back to sequential mode: %s", e,
                    )
                    self._st["_warned_batched"] = True
                    self._st["_disable_batched"] = True
        outs = []
        for artist_i in individuals:
            artist_b = broadcast_batch(artist_i, bsz).to(
                device=context.device, dtype=context.dtype)
            kv = torch.cat([context, artist_b], dim=1) \
                if fusion_mode == FUSION_CONCAT_WITH_BASE else artist_b
            out_i = self.original(q_x, kv, rope_emb=rope_emb, transformer_options=t_opts)
            outs.append(out_i)
        return outs

    def _artist_chunks(self, individuals):
        """Split artists into chunks of max_batch_artists (0 = no limit)."""
        limit = int(self._st.get("max_batch_artists", 0) or 0)
        if limit <= 0 or len(individuals) <= limit:
            return [individuals]
        return [
            individuals[i:i + limit]
            for i in range(0, len(individuals), limit)
        ]

    def _batched_chunk_forward(self, x, context, rope_emb, t_opts,
                               chunk, fusion_mode):
        """One batched forward over a chunk of artists. Returns (n, B, T, D)."""
        bsz = context.shape[0]
        n = len(chunk)
        kv_list = []
        for artist_i in chunk:
            artist_b = broadcast_batch(artist_i, bsz).to(
                device=context.device, dtype=context.dtype)
            if fusion_mode == FUSION_CONCAT_WITH_BASE:
                kv_list.append(torch.cat([context, artist_b], dim=1))
            else:
                kv_list.append(artist_b)
        kv_lens = {kv.shape[1] for kv in kv_list}
        if len(kv_lens) > 1:
            raise ValueError(f"K/V lengths differ {kv_lens}; cannot batch")
        x_rep = x.repeat(n, *([1] * (x.dim() - 1)))
        kv_stacked = torch.cat(kv_list, dim=0)
        rope_rep = rope_emb
        if rope_emb is not None and torch.is_tensor(rope_emb):
            if rope_emb.dim() > 0 and rope_emb.shape[0] == bsz:
                rope_rep = rope_emb.repeat(n, *([1] * (rope_emb.dim() - 1)))
        new_opts = dict(t_opts) if isinstance(t_opts, dict) else {}
        cou = new_opts.get("cond_or_uncond")
        if cou is not None:
            new_opts["cond_or_uncond"] = list(cou) * n
        out = self.original(x_rep, kv_stacked, rope_emb=rope_rep,
                            transformer_options=new_opts)
        return out.view(n, bsz, *out.shape[1:])

    def _batched_artists_outputs_only(self, x, context, rope_emb, t_opts,
                                      individuals, fusion_mode):
        """All artists' forwards batched (chunked); returns list of (B, T, D)."""
        outs = []
        for chunk in self._artist_chunks(individuals):
            stacked = self._batched_chunk_forward(
                x, context, rope_emb, t_opts, chunk, fusion_mode
            )
            outs.extend(stacked[i] for i in range(stacked.shape[0]))
        return outs

    def _batched_artists_forward(self, x, context, rope_emb, t_opts,
                                 individuals, weights, fusion_mode):
        """Weighted sum over batched artist forwards (chunked)."""
        total = None
        offset = 0
        for chunk in self._artist_chunks(individuals):
            stacked = self._batched_chunk_forward(
                x, context, rope_emb, t_opts, chunk, fusion_mode
            )
            n = stacked.shape[0]
            w_t = torch.tensor(
                weights[offset:offset + n],
                device=stacked.device, dtype=stacked.dtype,
            ).view(n, *([1] * (stacked.dim() - 1)))
            part = (stacked * w_t).sum(dim=0)
            total = part if total is None else total + part
            offset += n
        return total

    # ----------------------------------------------------------- lowrank path

    def _fwd_lowrank_avg(self, x, context, rope_emb, t_opts,
                          individuals, weights, fades, mask, fusion_mode, strength,
                          fp=None):
        """LoRA-style low-rank injection.

        delta_i = A_i - A_base
        D = stack(delta_i)              # (N, M)
        D_lowrank = topk_rowspace_project(D, k)
        delta_avg = sum(w_i * D_lowrank[i])
        artist_total = A_base + delta_avg

        Timing fades scale each artist's delta directly (the path is already
        delta-space, so a faded artist converges to the base naturally).
        """
        ws, _ = self._effective_weights(weights, fades)
        norm_scope = _resolve_norm_lock_scope(self._st.get("norm_lock_scope", NORM_LOCK_SCOPE_PER_ARTIST))
        do_norm_lock = self._st.get("match_base_norm", False) and fusion_mode in (
            FUSION_INTERPOLATE, FUSION_BASE_PRESERVE
        )
        per_artist_lock = do_norm_lock and norm_scope in (
            NORM_LOCK_SCOPE_PER_ARTIST, NORM_LOCK_SCOPE_BOTH
        )
        mixed_lock = do_norm_lock and norm_scope in (
            NORM_LOCK_SCOPE_MIXED, NORM_LOCK_SCOPE_BOTH
        )
        base_out = self.original(x, context, rope_emb=rope_emb, transformer_options=t_opts)
        active = [
            (artist, weight) for artist, weight in zip(individuals, ws)
            if abs(float(weight)) > 1e-8
        ]
        if not active:
            return base_out
        individuals = [item[0] for item in active]
        ws = [item[1] for item in active]
        n = len(individuals)
        k = int(self._st.get("lowrank_k", 1))
        k = max(1, min(k, n))
        balance_deltas = (
            self._contribution_balance_alpha() > 0.0
            and fusion_mode in (FUSION_INTERPOLATE, FUSION_BASE_PRESERVE)
            and n >= 2
        )
        artist_outs = self._get_artist_outputs_with_cache(
            x, context, rope_emb, t_opts, individuals, fusion_mode,
            base_out=base_out, fp=fp,
        )
        if per_artist_lock:
            artist_outs = [
                self._match_base_norm(o, base_out, mask, scale_floor=0.0)
                for o in artist_outs
            ]
        if balance_deltas:
            artist_outs = self._balance_artist_deltas(artist_outs, base_out, ws, mask)
        out_dtype = base_out.dtype

        A = torch.stack(artist_outs, dim=0).to(torch.float32)   # (N, B, T, D)
        base_f32 = base_out.to(torch.float32).unsqueeze(0)      # (1, B, T, D)
        delta = A - base_f32                                    # (N, B, T, D)

        orig_shape = delta.shape
        D_mat = delta.reshape(n, -1)                            # (N, M)

        if k < n:
            try:
                D_lowrank = lowrank_rows_deterministic(D_mat, k)
            except Exception as e:
                if not self._st.get("_warned_svd", False):
                    logger.warning(
                        "[AnimaCrossAttn] L%d lowrank_avg failed; this step "
                        "degrades to output_avg: %s", self._idx, e,
                    )
                    self._st["_warned_svd"] = True
                D_lowrank = D_mat
        else:
            # k >= n is mathematically output_avg (no projection).
            D_lowrank = D_mat

        w_t = torch.tensor(ws, device=D_lowrank.device, dtype=D_lowrank.dtype).view(n, 1)
        delta_avg = (D_lowrank * w_t).sum(dim=0)                # (M,)
        delta_avg = delta_avg.reshape(orig_shape[1:]).to(out_dtype)  # (B, T, D)

        artist_total = base_out + delta_avg

        artist_total = self._apply_ema(artist_total, fusion_mode, fp=fp)
        if mixed_lock:
            artist_total = self._match_base_norm(artist_total, base_out, mask)
        artist_total = self._cap_mixed_delta(
            artist_total, base_out, mask, fusion_mode, strength,
        )

        if fusion_mode == FUSION_INTERPOLATE and strength == 1.0 and all(mask):
            return artist_total
        return self._apply_fusion(base_out, artist_total, mask, fusion_mode, strength)

    # ---------------------------------------------------------- combined path

    def _fwd_with_combined(self, x, context, rope_emb, t_opts,
                           combined, mask, fusion_mode, strength,
                           fp=None, extra_fp=None):
        bsz = context.shape[0]
        artist_b = broadcast_batch(combined, bsz).to(
            device=context.device, dtype=context.dtype)

        norm_scope = _resolve_norm_lock_scope(self._st.get("norm_lock_scope", NORM_LOCK_SCOPE_PER_ARTIST))
        do_norm_lock = self._st.get("match_base_norm", False) and fusion_mode in (
            FUSION_INTERPOLATE, FUSION_BASE_PRESERVE
        )
        mixed_lock = do_norm_lock and norm_scope in (
            NORM_LOCK_SCOPE_MIXED, NORM_LOCK_SCOPE_BOTH
        )

        if fusion_mode in (FUSION_INTERPOLATE, FUSION_BASE_PRESERVE):
            base_out = self.original(x, context, rope_emb=rope_emb, transformer_options=t_opts)
            # Reuse the K-step averaging machinery with a single pseudo
            # artist, so combined paths get the same temporal smoothing as
            # output_avg instead of a first-step-only snapshot. extra_fp carries
            # the weight*fade so a freeze cannot lock a stale mid-fade weight.
            outs = self._get_artist_outputs_with_cache(
                x, context, rope_emb, t_opts, [artist_b], fusion_mode,
                base_out=base_out, extra_fp=extra_fp, fp=fp,
            )
            artist_out = outs[0]
            artist_out = self._apply_ema(artist_out, fusion_mode, fp=fp)
            if mixed_lock:
                artist_out = self._match_base_norm(artist_out, base_out, mask)
            artist_out = self._cap_mixed_delta(
                artist_out, base_out, mask, fusion_mode, strength,
            )

            if fusion_mode == FUSION_INTERPOLATE and strength == 1.0 and all(mask):
                return artist_out
            return self._apply_fusion(base_out, artist_out, mask, fusion_mode, strength)

        # FUSION_CONCAT_WITH_BASE: every masked-in row gets the artist tokens
        # appended to its K/V. Padding uncond rows with zero tokens (the old
        # behavior) still fed them softmax weight and diluted the CFG uncond
        # output. Instead, when only some rows are masked in, run the base
        # forward too and select per row so uncond stays exactly the base.
        merged = torch.cat([context, artist_b], dim=1)
        if all(mask):
            return self.original(x, merged, rope_emb=rope_emb, transformer_options=t_opts)
        merged_out = self.original(x, merged, rope_emb=rope_emb, transformer_options=t_opts)
        base_out = self.original(x, context, rope_emb=rope_emb, transformer_options=t_opts)
        row_mask = _row_mask_like(mask, merged_out)
        return torch.where(row_mask, merged_out, base_out)
