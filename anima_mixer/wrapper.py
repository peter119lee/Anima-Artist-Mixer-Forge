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
    STATIC_CAPTURE_K_DEFAULT,
    ANCHOR_LAYER_THRESHOLD_DISABLED,
)
from .math_utils import (
    lowrank_rows_deterministic,
    project_perpendicular,
    timing_fade_factor,
)
from .parsing import normalize_weights
from .patching import broadcast_batch, build_artists, in_sigma_range, resolve_mask

logger = logging.getLogger(__name__)


def _combine_concat(individuals, weights):
    parts = [a * float(w) for a, w in zip(individuals, weights)]
    return torch.cat(parts, dim=1)


def _cache_store(tensor, low_vram):
    """Detach a tensor for caching, optionally offloading to CPU."""
    t = tensor.detach()
    return t.cpu() if low_vram else t


def _cache_load(tensor, like):
    """Bring a cached tensor back to the compute device/dtype of ``like``."""
    return tensor.to(device=like.device, dtype=like.dtype)


class CrossAttnWrapper(nn.Module):
    def __init__(self, original, shared_state, layer_idx):
        super().__init__()
        self.original = original
        self._st = shared_state
        self._idx = layer_idx
        self._disabled = False

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

    def _apply_ema(self, artist_total, fusion_mode):
        """Cross-step EMA smoothing (fusion in {interpolate, base_preserve}).

        concat_with_base never produces an artist_total, and static capture
        already freezes artist outputs, so EMA is skipped in both cases.
        """
        if self._st.get("artist_static_capture", False):
            return artist_total
        ema_alpha = float(self._st.get("artist_ema_alpha", 0.0))
        ema_compatible = fusion_mode in (FUSION_INTERPOLATE, FUSION_BASE_PRESERVE)
        if ema_alpha <= 0.0 or not ema_compatible:
            return artist_total
        self._maybe_reset_ema()
        cache = self._st.setdefault("_ema_cache", {})
        prev = cache.get(self._idx)
        if prev is not None and prev.shape == artist_total.shape:
            artist_total = ema_alpha * prev + (1.0 - ema_alpha) * artist_total
        cache[self._idx] = artist_total.detach()
        return artist_total

    # -------------------------------------------------------- static capture

    def _maybe_reset_static(self):
        """Reset the static cache when a new sampling run starts.

        Within one run sigma decreases monotonically, so this never fires.
        Across runs the first step jumps sigma back up -> reset. CFG double
        forwards repeat the same sigma -> no reset.
        """
        st = self._st
        cur = st.get("current_sigma")
        if cur is None:
            return
        prev_max = st.get("_static_max_sigma")
        if prev_max is None or cur > prev_max + 1e-3:
            st["_static_cache"] = {}
            st["_static_max_sigma"] = cur

    def _get_artist_outputs_with_cache(self, x, context, rope_emb, t_opts,
                                       individuals, fusion_mode):
        """H' temporal averaging: accumulate the first K steps, then freeze.

        Accumulation runs in fp32; returned tensors keep the model dtype.
        Cache fingerprint = (x.shape, n); resolution or artist-count changes
        invalidate the entry. A sigma jump (new run) resets everything.
        Repeated calls at the same sigma (CFG double forward) reuse the
        current average without re-accumulating.
        """
        st = self._st
        low_vram = bool(st.get("low_vram_cache", False))
        if not st.get("artist_static_capture", False):
            return self._collect_artist_outputs(
                x, context, rope_emb, t_opts, individuals, fusion_mode
            )
        # static capture cannot work for concat_with_base (x changes every
        # step, and the artist attention includes the base context).
        if fusion_mode == FUSION_CONCAT_WITH_BASE:
            return self._collect_artist_outputs(
                x, context, rope_emb, t_opts, individuals, fusion_mode
            )

        self._maybe_reset_static()
        cache = st.setdefault("_static_cache", {})
        n = len(individuals)
        fp = (tuple(x.shape), n)

        cur_sigma = st.get("current_sigma")
        sigma_key = round(float(cur_sigma), 4) if cur_sigma is not None else None

        entry = cache.get(self._idx)
        if entry is None or entry.get("_fp") != fp:
            entry = {
                "_fp": fp,
                "seen_sigmas": set(),
                "accumulator": None,
                "count": 0,
                "frozen": False,
                "frozen_outputs": None,
            }
            cache[self._idx] = entry

        if entry["frozen"]:
            return [_cache_load(o, context) for o in entry["frozen_outputs"]]

        # Same sigma re-entry (CFG second forward): return the current
        # average without recomputing or re-accumulating.
        if sigma_key is not None and sigma_key in entry["seen_sigmas"]:
            if entry["accumulator"] is not None and entry["count"] > 0:
                inv = 1.0 / entry["count"]
                return [
                    _cache_load(a * inv, context) for a in entry["accumulator"]
                ]
            return self._collect_artist_outputs(
                x, context, rope_emb, t_opts, individuals, fusion_mode
            )

        outs = self._collect_artist_outputs(
            x, context, rope_emb, t_opts, individuals, fusion_mode
        )
        if entry["accumulator"] is None:
            entry["accumulator"] = [
                _cache_store(o.to(torch.float32), low_vram) for o in outs
            ]
        else:
            for i, o in enumerate(outs):
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
            return [_cache_load(o, context) for o in entry["frozen_outputs"]]

        inv = 1.0 / entry["count"]
        return [_cache_load(a * inv, context) for a in entry["accumulator"]]

    # ----------------------------------------------------------------- fusion

    def _apply_fusion(self, base_out, artist_total, mask, fusion_mode, strength):
        """Single fusion exit for interpolate and base_preserve.

        concat_with_base never reaches this point (handled in
        _fwd_with_combined).
        """
        if fusion_mode == FUSION_BASE_PRESERVE:
            delta = artist_total - base_out
            delta_perp = project_perpendicular(delta, base_out)
            out = base_out.clone()
            for i, hit in enumerate(mask):
                if hit:
                    out[i] = base_out[i] + strength * delta_perp[i]
            return out

        out = base_out.clone()
        for i, hit in enumerate(mask):
            if hit:
                out[i] = base_out[i] * (1.0 - strength) + artist_total[i] * strength
        return out

    def _match_base_norm(self, artist_total, base_out, mask):
        """Rescale the mixed artist output to the base output's RMS energy.

        The weighted artist mixture can carry noticeably different
        activation energy than the base output downstream blocks were
        trained on; the deviation compounds across layers and surfaces as
        seed-dependent style-strength swings (style drift). Per-row RMS
        matching keeps the artist direction (the style) while restoring
        on-distribution magnitude. The scale is clamped to [0.5, 2.0] so
        pathological mismatches degrade gracefully instead of
        overcorrecting. Rows outside the injection mask keep scale 1.
        """
        dims = tuple(range(1, artist_total.dim()))
        base_rms = base_out.detach().to(torch.float32).pow(2).mean(
            dim=dims, keepdim=True).sqrt()
        artist_rms = artist_total.detach().to(torch.float32).pow(2).mean(
            dim=dims, keepdim=True).sqrt()
        scale = (base_rms / artist_rms.clamp(min=1e-6)).clamp(0.5, 2.0)
        for i, hit in enumerate(mask):
            if not hit:
                scale[i] = 1.0
        return artist_total * scale.to(artist_total.dtype)

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
            return self.original(x, context, rope_emb=rope_emb,
                                 transformer_options=transformer_options)

        if not st.get("enabled", False) or context is None:
            return self.original(x, context, rope_emb=rope_emb,
                                 transformer_options=transformer_options)

        if self._disabled:
            return self.original(x, context, rope_emb=rope_emb,
                                 transformer_options=transformer_options)

        if not in_sigma_range(st):
            return self.original(x, context, rope_emb=rope_emb,
                                 transformer_options=transformer_options)

        try:
            return self._dispatch(x, context, rope_emb, transformer_options)
        except Exception as e:
            logger.exception(
                "[AnimaCrossAttn] L%d injection failed; this layer falls back "
                "to the original cross-attention: %s", self._idx, e,
            )
            self._disabled = True
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
            for artist, weight, route, timing in zip(
                individuals, weights, routes, timings,
            ):
                if route is not None and self._idx not in route:
                    continue
                fade = timing_fade_factor(timing, cur_sigma)
                if fade <= 0.0:
                    continue
                filtered.append((artist, weight, fade))
            if not filtered:
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

        # lowrank_avg is meaningless for a single artist (no multi-artist
        # directions to project); it degrades to output_avg below.
        if combine_mode == COMBINE_LOWRANK_AVG and len(individuals) >= 2:
            return self._fwd_lowrank_avg(
                x, context, rope_emb, transformer_options,
                individuals, weights, fades, mask, fusion_mode, strength,
            )

        if combine_mode in (COMBINE_OUTPUT_AVG, COMBINE_LOWRANK_AVG):
            return self._fwd_output_avg(
                x, context, rope_emb, transformer_options,
                individuals, weights, fades, mask, fusion_mode, strength,
            )

        # concat never normalizes, so the fade multiplies the raw weight.
        combined = _combine_concat(
            individuals, [w * f for w, f in zip(weights, fades)],
        )
        return self._fwd_with_combined(
            x, context, rope_emb, transformer_options,
            combined, mask, fusion_mode, strength,
        )

    def _effective_weights(self, weights, fades):
        """Resolve normalization and timing fades into final weights.

        Normalization runs on the raw weights FIRST, then each artist's
        share is scaled by its fade factor — otherwise normalizing after the
        fade would cancel it whenever a layer has a single active artist
        (the common layer_scheduled case). Returns ``(ws, fade_comp)`` where
        ``fade_comp = sum(w_norm * (1 - fade))`` is the share of weight that
        faded out and should be returned to the base output so a fully faded
        artist converges to the original cross-attention instead of zero.
        """
        if self._st.get("normalize_weights", True):
            ws_base = normalize_weights(weights)
        else:
            ws_base = list(weights)
        ws = [w * f for w, f in zip(ws_base, fades)]
        fade_comp = sum(w * (1.0 - f) for w, f in zip(ws_base, fades))
        return ws, fade_comp

    def _fwd_output_avg(self, x, context, rope_emb, t_opts,
                        individuals, weights, fades, mask, fusion_mode, strength):
        bsz = context.shape[0]

        ws, fade_comp = self._effective_weights(weights, fades)
        n = len(individuals)
        static_capture = self._st.get("artist_static_capture", False)
        # The static-capture path must collect N independent outputs to cache
        # them. concat_with_base cannot be cached and skips this.
        force_collect = static_capture and fusion_mode != FUSION_CONCAT_WITH_BASE

        artist_total = None
        if force_collect:
            outs = self._get_artist_outputs_with_cache(
                x, context, rope_emb, t_opts, individuals, fusion_mode
            )
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
                artist_total = out_i * w if artist_total is None else artist_total + out_i * w

        base_out = None
        if abs(fade_comp) > 1e-6:
            # Return the faded-out weight share to the base output so a fully
            # faded artist converges to the original cross-attention.
            base_out = self.original(x, context, rope_emb=rope_emb, transformer_options=t_opts)
            artist_total = artist_total + fade_comp * base_out

        artist_total = self._apply_ema(artist_total, fusion_mode)

        match_norm = (
            self._st.get("match_base_norm", True)
            and fusion_mode in (FUSION_INTERPOLATE, FUSION_BASE_PRESERVE)
        )
        # base_preserve always needs base_out for the projection. interpolate
        # can skip it only at exactly strength == 1.0 (extrapolation beyond
        # 1.0 starts from base again) — unless norm matching needs the base
        # reference anyway.
        skip_fusion = (
            fusion_mode == FUSION_INTERPOLATE and strength == 1.0 and all(mask)
        )
        if base_out is None and (match_norm or not skip_fusion):
            base_out = self.original(x, context, rope_emb=rope_emb, transformer_options=t_opts)
        if match_norm:
            artist_total = self._match_base_norm(artist_total, base_out, mask)
        if skip_fusion:
            return artist_total
        return self._apply_fusion(base_out, artist_total, mask, fusion_mode, strength)

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
                         individuals, weights, fades, mask, fusion_mode, strength):
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
        n = len(individuals)
        k = int(self._st.get("lowrank_k", 1))
        k = max(1, min(k, n))

        artist_outs = self._get_artist_outputs_with_cache(
            x, context, rope_emb, t_opts, individuals, fusion_mode
        )

        base_out = self.original(x, context, rope_emb=rope_emb, transformer_options=t_opts)
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

        artist_total = self._apply_ema(artist_total, fusion_mode)

        if fusion_mode == FUSION_INTERPOLATE and strength == 1.0 and all(mask):
            return artist_total
        return self._apply_fusion(base_out, artist_total, mask, fusion_mode, strength)

    # ---------------------------------------------------------- combined path

    def _fwd_with_combined(self, x, context, rope_emb, t_opts,
                           combined, mask, fusion_mode, strength):
        bsz = context.shape[0]
        artist_b = broadcast_batch(combined, bsz).to(
            device=context.device, dtype=context.dtype)

        if fusion_mode in (FUSION_INTERPOLATE, FUSION_BASE_PRESERVE):
            base_out = self.original(x, context, rope_emb=rope_emb, transformer_options=t_opts)
            # Reuse the K-step averaging machinery with a single pseudo
            # artist, so combined paths get the same temporal smoothing as
            # output_avg instead of a first-step-only snapshot.
            outs = self._get_artist_outputs_with_cache(
                x, context, rope_emb, t_opts, [artist_b], fusion_mode
            )
            artist_out = outs[0]
            artist_out = self._apply_ema(artist_out, fusion_mode)
            if self._st.get("match_base_norm", True):
                artist_out = self._match_base_norm(artist_out, base_out, mask)

            if fusion_mode == FUSION_INTERPOLATE and strength == 1.0 and all(mask):
                return artist_out
            return self._apply_fusion(base_out, artist_out, mask, fusion_mode, strength)

        # FUSION_CONCAT_WITH_BASE
        artist_len = artist_b.shape[1]
        extension = torch.zeros(bsz, artist_len, context.shape[-1],
                                device=context.device, dtype=context.dtype)
        for i, hit in enumerate(mask):
            if hit:
                extension[i] = artist_b[i]
        merged = torch.cat([context, extension], dim=1)
        return self.original(x, merged, rope_emb=rope_emb, transformer_options=t_opts)
