import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

FUSION_INTERPOLATE = "interpolate"
FUSION_CONCAT_WITH_BASE = "concat_with_base"

COMBINE_CONCAT = "concat"
COMBINE_OUTPUT_AVG = "output_avg"

MAX_ARTISTS = 32


def _extract(conditioning):
    if conditioning is None:
        return None, None, None
    if not isinstance(conditioning, (list, tuple)) or len(conditioning) == 0:
        return None, None, None
    first = conditioning[0]
    if not isinstance(first, (list, tuple)) or len(first) == 0:
        return None, None, None
    raw = first[0] if torch.is_tensor(first[0]) else None
    extra = first[1] if len(first) > 1 and isinstance(first[1], dict) else {}
    return raw, extra.get("t5xxl_ids"), extra.get("t5xxl_weights")


def _split_artist_chain(chain):
    if not chain:
        return []
    s = str(chain).replace("，", ",").replace("\n", ",").replace("\r", ",")
    parts = [p.strip() for p in s.split(",")]
    return [p for p in parts if p]


def _parse_layer_filter(text, num_blocks):
    if not text:
        return None
    s = str(text).replace("，", ",").replace(" ", "")
    if not s:
        return None
    result = set()
    for part in s.split(","):
        if not part:
            continue
        if "-" in part[1:]:
            dash_idx = part.index("-", 1)
            try:
                lo = int(part[:dash_idx])
                hi = int(part[dash_idx + 1:])
            except ValueError:
                continue
            if lo < 0:
                lo += num_blocks
            if hi < 0:
                hi += num_blocks
            if lo > hi:
                lo, hi = hi, lo
            lo = max(0, lo)
            hi = min(num_blocks - 1, hi)
            if lo <= hi:
                result.update(range(lo, hi + 1))
        else:
            try:
                v = int(part)
            except ValueError:
                continue
            if v < 0:
                v += num_blocks
            if 0 <= v < num_blocks:
                result.add(v)
    return sorted(result) if result else None


def _normalize_weights(weights):
    total = sum(abs(w) for w in weights)
    if total <= 1e-8:
        return [1.0 / len(weights)] * len(weights)
    return [w / total for w in weights]


def _unwrap_cross_attn(ca):
    while isinstance(ca, _CrossAttnWrapper):
        ca = ca.original
    return ca


def _validate(diffusion_model):
    if not hasattr(diffusion_model, "blocks"):
        return False, 0, 0, f"{type(diffusion_model).__name__} 没有 .blocks"
    blocks = diffusion_model.blocks
    if len(blocks) == 0:
        return False, 0, 0, ".blocks 为空"
    b0 = blocks[0]
    if not hasattr(b0, "cross_attn"):
        return False, 0, 0, "blocks[0] 没有 cross_attn"
    ca = _unwrap_cross_attn(b0.cross_attn)
    if not hasattr(ca, "context_dim"):
        return False, 0, 0, "cross_attn 没有 context_dim"
    return True, len(blocks), int(ca.context_dim), "ok"


def _cleanup_residual_wrappers(dm):
    if not hasattr(dm, "blocks"):
        return 0
    cleaned = 0
    for i in range(len(dm.blocks)):
        blk = dm.blocks[i]
        if not hasattr(blk, "cross_attn"):
            continue
        original = _unwrap_cross_attn(blk.cross_attn)
        if blk.cross_attn is not original:
            blk.cross_attn = original
            cleaned += 1
    return cleaned


def _preprocess_one(dm, raw, ids, weights, target_device, target_dtype):
    if ids is None:
        artist = raw.to(device=target_device, dtype=target_dtype)
        if artist.dim() == 2:
            artist = artist.unsqueeze(0)
        return artist
    raw_b = raw if raw.dim() == 3 else raw.unsqueeze(0)
    ids_b = ids if ids.dim() >= 2 else ids.unsqueeze(0)
    weights_b = None
    if weights is not None:
        if weights.dim() == 1:
            weights_b = weights.unsqueeze(0).unsqueeze(-1)
        elif weights.dim() == 2:
            weights_b = weights.unsqueeze(-1)
        else:
            weights_b = weights
    raw_b = raw_b.to(device=target_device, dtype=target_dtype)
    ids_b = ids_b.to(device=target_device)
    if weights_b is not None:
        weights_b = weights_b.to(device=target_device, dtype=target_dtype)
    with torch.inference_mode():
        return dm.preprocess_text_embeds(raw_b, ids_b, t5xxl_weights=weights_b)


def _build_artists(state, ref_context):
    if state.get("individuals") is not None:
        return state["individuals"], state["real_lens"]
    dm = state["dm_ref"]
    individuals, real_lens = [], []
    for raw, ids, w_t in zip(state["raws"], state["ids_list"], state["w_list"]):
        artist = _preprocess_one(dm, raw, ids, w_t, ref_context.device, ref_context.dtype)
        individuals.append(artist)
        real_lens.append(int(ids.shape[-1]) if ids is not None else artist.shape[1])
    state["individuals"] = individuals
    state["real_lens"] = real_lens
    return individuals, real_lens


def _combine_concat(individuals, weights):
    parts = [a * float(w) for a, w in zip(individuals, weights)]
    return torch.cat(parts, dim=1)


def _broadcast_batch(t, batch_size):
    if t.shape[0] == batch_size:
        return t
    if t.shape[0] == 1:
        return t.expand(batch_size, -1, -1)
    if batch_size % t.shape[0] == 0:
        return t.repeat(batch_size // t.shape[0], 1, 1)
    return t[:1].expand(batch_size, -1, -1)


def _resolve_mask(cou, batch_size, apply_to_uncond, state):
    if cou is None or len(cou) != batch_size:
        if not state.get("_warned", False):
            logger.warning(
                "[AnimaCrossAttn] 未拿到 cond_or_uncond (got=%s, batch=%d)，"
                "退化为对所有行注入。", cou, batch_size,
            )
            state["_warned"] = True
        return [True] * batch_size
    if apply_to_uncond:
        return [True] * batch_size
    return [c == 0 for c in cou]


def _in_sigma_range(state):
    rng = state.get("sigma_range")
    if rng is None:
        return True
    cur = state.get("current_sigma")
    if cur is None:
        return True
    lo, hi = rng
    return lo <= cur <= hi


class _CrossAttnWrapper(nn.Module):
    def __init__(self, original, shared_state, layer_idx):
        super().__init__()
        self.original = original
        self._st = shared_state
        self._idx = layer_idx
        self._disabled = False

    def forward(self, x, context=None, rope_emb=None, transformer_options={}):
        st = self._st
        if not st.get("enabled", False) or context is None:
            return self.original(x, context, rope_emb=rope_emb,
                                 transformer_options=transformer_options)

        if self._disabled:
            return self.original(x, context, rope_emb=rope_emb,
                                 transformer_options=transformer_options)

        if not _in_sigma_range(st):
            return self.original(x, context, rope_emb=rope_emb,
                                 transformer_options=transformer_options)

        try:
            return self._dispatch(x, context, rope_emb, transformer_options)
        except Exception as e:
            logger.exception(
                "[AnimaCrossAttn] L%d 注入路径异常，本层退化为原始 cross_attn: %s",
                self._idx, e,
            )
            self._disabled = True
            return self.original(x, context, rope_emb=rope_emb,
                                 transformer_options=transformer_options)

    def _dispatch(self, x, context, rope_emb, transformer_options):
        st = self._st
        individuals, _ = _build_artists(st, context)
        combine_mode = st["combine_mode"]
        fusion_mode = st["fusion_mode"]
        strength = float(st["strength"])
        weights = st["user_weights"]

        cou = transformer_options.get("cond_or_uncond") if isinstance(transformer_options, dict) else None
        bsz = context.shape[0]
        mask = _resolve_mask(cou, bsz, st["apply_to_uncond"], st)

        if not any(mask):
            return self.original(x, context, rope_emb=rope_emb,
                                 transformer_options=transformer_options)

        if combine_mode == COMBINE_OUTPUT_AVG:
            return self._fwd_output_avg(
                x, context, rope_emb, transformer_options,
                individuals, weights, mask, fusion_mode, strength,
            )

        combined = _combine_concat(individuals, weights)
        return self._fwd_with_combined(
            x, context, rope_emb, transformer_options,
            combined, mask, fusion_mode, strength,
        )

    def _fwd_output_avg(self, x, context, rope_emb, t_opts,
                       individuals, weights, mask, fusion_mode, strength):
        bsz = context.shape[0]
        if self._st.get("normalize_weights", True):
            ws = _normalize_weights(weights)
        else:
            ws = list(weights)
        n = len(individuals)
        artist_total = None
        if n >= 2 and not self._st.get("_disable_batched", False):
            try:
                artist_total = self._batched_artists_forward(
                    x, context, rope_emb, t_opts, individuals, ws, fusion_mode
                )
            except Exception as e:
                if not self._st.get("_warned_batched", False):
                    logger.warning(
                        "[AnimaCrossAttn] batched output_avg 失败，回退到串行模式: %s", e,
                    )
                    self._st["_warned_batched"] = True
                    self._st["_disable_batched"] = True
                artist_total = None
        if artist_total is None:
            for artist_i, w in zip(individuals, ws):
                artist_b = _broadcast_batch(artist_i, bsz).to(
                    device=context.device, dtype=context.dtype)
                kv = torch.cat([context, artist_b], dim=1) \
                    if fusion_mode == FUSION_CONCAT_WITH_BASE else artist_b
                out_i = self.original(x, kv, rope_emb=rope_emb, transformer_options=t_opts)
                artist_total = out_i * w if artist_total is None else artist_total + out_i * w
        if strength >= 1.0 and all(mask):
            return artist_total
        base_out = self.original(x, context, rope_emb=rope_emb, transformer_options=t_opts)
        out = base_out.clone()
        for i, hit in enumerate(mask):
            if hit:
                out[i] = base_out[i] * (1.0 - strength) + artist_total[i] * strength
        return out
    def _batched_artists_forward(self, x, context, rope_emb, t_opts,
                                 individuals, weights, fusion_mode):
        """N 个画师的 cross-attn forward 在 batch 维度并行。"""
        bsz = context.shape[0]
        n = len(individuals)
        kv_list = []
        for artist_i in individuals:
            artist_b = _broadcast_batch(artist_i, bsz).to(
                device=context.device, dtype=context.dtype)
            if fusion_mode == FUSION_CONCAT_WITH_BASE:
                kv_list.append(torch.cat([context, artist_b], dim=1))
            else:
                kv_list.append(artist_b)
        kv_lens = {kv.shape[1] for kv in kv_list}
        if len(kv_lens) > 1:
            raise ValueError(f"K/V 长度不一致 {kv_lens}，无法 batch 化")
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
        out = out.view(n, bsz, *out.shape[1:])
        w_t = torch.tensor(weights, device=out.device, dtype=out.dtype).view(
            n, *([1] * (out.dim() - 1))
        )
        return (out * w_t).sum(dim=0)

    def _fwd_with_combined(self, x, context, rope_emb, t_opts,
                          combined, mask, fusion_mode, strength):
        bsz = context.shape[0]
        artist_b = _broadcast_batch(combined, bsz).to(
            device=context.device, dtype=context.dtype)

        if fusion_mode == FUSION_INTERPOLATE:
            base_out = self.original(x, context, rope_emb=rope_emb, transformer_options=t_opts)
            artist_out = self.original(x, artist_b, rope_emb=rope_emb, transformer_options=t_opts)
            out = base_out.clone()
            for i, hit in enumerate(mask):
                if hit:
                    out[i] = base_out[i] * (1.0 - strength) + artist_out[i] * strength
            return out

        artist_len = artist_b.shape[1]
        extension = torch.zeros(bsz, artist_len, context.shape[-1],
                                device=context.device, dtype=context.dtype)
        for i, hit in enumerate(mask):
            if hit:
                extension[i] = artist_b[i]
        merged = torch.cat([context, extension], dim=1)
        return self.original(x, merged, rope_emb=rope_emb, transformer_options=t_opts)


def _make_sigma_capture(state, prev_wrapper):
    def wrapper(apply_model, options):
        ts = options.get("timestep")
        if ts is not None:
            try:
                state["current_sigma"] = float(ts.flatten()[0].item())
            except Exception:
                pass
        if prev_wrapper is not None:
            return prev_wrapper(apply_model, options)
        return apply_model(options["input"], options["timestep"], **options["c"])
    return wrapper


class AnimaArtistPack:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "artist_chain": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "画师串。用英文/中文逗号或换行分隔，例: wlop, sakimichan, krenz。"
                                "支持权重语法 (wlop:1.1)"
                }),
            },
            "optional": {
                "base_prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "主词条（可选）。按 Anima 推荐写法：画师在前，换行后跟主词条，"
                                "即 '<artist>\\n<base_prompt>'。留空则只编码画师名本身"
                }),
            },
        }

    RETURN_TYPES = ("ANIMA_PACK",)
    RETURN_NAMES = ("artist_pack",)
    FUNCTION = "pack"
    CATEGORY = "Anima/CrossAttn"

    def pack(self, clip, artist_chain, base_prompt=""):
        names = _split_artist_chain(artist_chain)
        base = (base_prompt or "").strip()

        try:
            base_tokens = clip.tokenize(base)
            base_conditioning = clip.encode_from_tokens_scheduled(base_tokens)
        except Exception as e:
            raise ValueError(
                f"[AnimaArtistPack] base_prompt 编码失败 (text={base!r}): {e}"
            )

        if not names:
            return ({
                "conditionings": [],
                "labels": [],
                "base_prompt": base,
                "base_conditioning": base_conditioning,
            },)

        if len(names) > MAX_ARTISTS:
            logger.warning(
                "[AnimaArtistPack] 画师数 %d 超过上限 %d，截断",
                len(names), MAX_ARTISTS,
            )
            names = names[:MAX_ARTISTS]

        conditionings = []
        for name in names:
            text = f"{name}\n{base}" if base else name
            try:
                tokens = clip.tokenize(text)
                cond = clip.encode_from_tokens_scheduled(tokens)
            except Exception as e:
                raise ValueError(
                    f"[AnimaArtistPack] 编码失败 (text={text!r}): {e}"
                )
            conditionings.append(cond)

        return ({
            "conditionings": conditionings,
            "labels": names,
            "base_prompt": base,
            "base_conditioning": base_conditioning,
        },)


class AnimaArtistOptions:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "start_block": ("INT", {
                    "default": 0, "min": 0, "max": 63, "step": 1,
                    "tooltip": "起始 block（含）。0 = 第一层"
                }),
                "end_block": ("INT", {
                    "default": -1, "min": -1, "max": 63, "step": 1,
                    "tooltip": "终止 block（含）。-1 = 最后一层"
                }),
                "start_percent": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001,
                    "tooltip": "采样进度起点。0.0 = 采样开始"
                }),
                "end_percent": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001,
                    "tooltip": "采样进度终点。1.0 = 采样结束"
                }),
                "normalize_weights": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "True: weights 归一化为相对比例。False: weights 直接作为独立强度"
                }),
            },
            "optional": {
                "layer_filter": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "高级层选择（可选）。按逗号隔开的块索引串，支持区间和负索引。\n"
                                "例: '0,3,5-10,-1' = 第0、3、第5~10、最后一层。\n"
                                "填了该字段会覆盖 start_block/end_block。留空 = 不生效"
                }),
            },
        }

    RETURN_TYPES = ("ANIMA_OPTS",)
    RETURN_NAMES = ("advanced_options",)
    FUNCTION = "build"
    CATEGORY = "Anima/CrossAttn"

    def build(self, start_block, end_block, start_percent, end_percent, normalize_weights,
              layer_filter=""):
        return ({
            "start_block": int(start_block),
            "end_block": int(end_block),
            "start_percent": float(start_percent),
            "end_percent": float(end_percent),
            "normalize_weights": bool(normalize_weights),
            "layer_filter": str(layer_filter or ""),
        },)


class AnimaArtistCrossAttn:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "artist_pack": ("ANIMA_PACK",),
                "combine_mode": (
                    [COMBINE_CONCAT, COMBINE_OUTPUT_AVG],
                    {"default": COMBINE_OUTPUT_AVG},
                ),
                "fusion_mode": (
                    [FUSION_INTERPOLATE, FUSION_CONCAT_WITH_BASE],
                    {"default": FUSION_INTERPOLATE},
                ),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "enabled": ("BOOLEAN", {"default": True}),
                "apply_to_uncond": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "advanced_options": ("ANIMA_OPTS",),
            },
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING")
    RETURN_NAMES = ("model", "base_prompt")
    FUNCTION = "patch"
    CATEGORY = "Anima/CrossAttn"

    def patch(self, model, artist_pack, combine_mode, fusion_mode,
              strength, enabled, apply_to_uncond, advanced_options=None):
        adv = advanced_options or {}
        sb = int(adv.get("start_block", 0))
        eb = int(adv.get("end_block", -1))
        start_percent = float(adv.get("start_percent", 0.0))
        end_percent = float(adv.get("end_percent", 1.0))
        normalize_w = bool(adv.get("normalize_weights", True))
        layer_filter_text = str(adv.get("layer_filter", "") or "")
        use_sigma_range = (start_percent > 0.0) or (end_percent < 1.0)

        if not isinstance(artist_pack, dict):
            raise ValueError(
                "[AnimaCrossAttn] artist_pack 类型错误，请用 AnimaArtistPack 节点输出连接"
            )
        conditionings = artist_pack.get("conditionings") or []
        labels = artist_pack.get("labels") or []

        base_cond_out = artist_pack.get("base_conditioning")
        if base_cond_out is None:
            raise ValueError(
                "[AnimaCrossAttn] artist_pack 缺少 base_conditioning 字段。"
                "请重启 ComfyUI 让 AnimaArtistPack 重新加载到最新版本"
            )

        if not conditionings:
            return (model, base_cond_out)

        raws, ids_list, w_list = [], [], []
        for idx, c in enumerate(conditionings):
            raw, ids, w = _extract(c)
            if raw is None:
                label = labels[idx] if idx < len(labels) else f"#{idx}"
                raise ValueError(
                    f"[AnimaCrossAttn] artist[{label}] conditioning 为空。"
                    "clip 与 model 是否匹配？"
                )
            raws.append(raw)
            ids_list.append(ids)
            w_list.append(w)

        n = len(raws)
        user_weights = [1.0] * n

        if not normalize_w and n > 1 and combine_mode == COMBINE_OUTPUT_AVG:
            if n >= 4:
                raise ValueError(
                    f"[AnimaCrossAttn] normalize_weights=False 且画师数={n} (有效权重和也是 {n}，"
                    f"远超合理范围)。当前 combine=output_avg 下这会使 cross-attn 输出被放大 ~{n} 倍，"
                    f"几乎必崩。\n"
                    f"改进方案 (任选一)：\n"
                    f"  1) AnimaArtistOptions 里把 normalize_weights 改为 True（推荐）\n"
                    f"  2) AnimaArtistPack 的 artist_chain 中用 (name:0.3) 内联调低单个强度\n"
                    f"  3) combine_mode 改用 concat (不走加权和逻辑)"
                )
            elif n >= 2:
                logger.warning(
                    "[AnimaCrossAttn] normalize_weights=False 且画师数=%d (combine=output_avg)，"
                    "cross-attn 输出会被放大 ~%d 倍，可能过曝。如出问题请改用 "
                    "normalize_weights=True 或 combine=concat。",
                    n, n,
                )

        try:
            dm = model.get_model_object("diffusion_model")
        except Exception:
            dm = model.model.diffusion_model

        _cleanup_residual_wrappers(dm)

        ok, num_blocks, ctx_dim, msg = _validate(dm)
        if not ok:
            raise ValueError(f"[AnimaCrossAttn] 不支持的模型: {msg}")
        if not hasattr(dm, "preprocess_text_embeds"):
            raise ValueError("[AnimaCrossAttn] 不是 Anima")

        explicit_blocks = _parse_layer_filter(layer_filter_text, num_blocks)
        if explicit_blocks is not None:
            target_blocks = explicit_blocks
            sb_real, eb_real = target_blocks[0], target_blocks[-1]
        else:
            sb_real = max(0, sb)
            eb_real = num_blocks - 1 if eb < 0 else min(num_blocks - 1, eb)
            if sb_real > eb_real:
                raise ValueError(
                    f"[AnimaCrossAttn] start_block={sb_real} > end_block={eb_real} (共 {num_blocks})"
                )
            target_blocks = list(range(sb_real, eb_real + 1))

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
                    "[AnimaCrossAttn] 解析 sigma 范围失败: %s。时间步控制不生效", e
                )
                sigma_range = None

        m = model.clone()

        state = {
            "enabled": bool(enabled),
            "fusion_mode": fusion_mode,
            "combine_mode": combine_mode,
            "strength": float(strength),
            "apply_to_uncond": bool(apply_to_uncond),
            "raws": raws,
            "ids_list": ids_list,
            "w_list": w_list,
            "user_weights": user_weights,
            "normalize_weights": normalize_w,
            "individuals": None,
            "real_lens": None,
            "dm_ref": dm,
            "sigma_range": sigma_range,
            "current_sigma": None,
        }

        if sigma_range is not None:
            prev = m.model_options.get("model_function_wrapper")
            m.set_model_unet_function_wrapper(_make_sigma_capture(state, prev))

        for i in target_blocks:
            inner = _unwrap_cross_attn(dm.blocks[i].cross_attn)
            wrapper = _CrossAttnWrapper(inner, state, i)
            m.add_object_patch(f"diffusion_model.blocks.{i}.cross_attn", wrapper)

        return (m, base_cond_out)


NODE_CLASS_MAPPINGS = {
    "AnimaArtistPack": AnimaArtistPack,
    "AnimaArtistCrossAttn": AnimaArtistCrossAttn,
    "AnimaArtistOptions": AnimaArtistOptions,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AnimaArtistPack": "Anima Artist Pack (Split + Encode)",
    "AnimaArtistCrossAttn": "Anima Artist Cross-Attn",
    "AnimaArtistOptions": "Anima Artist Options (Advanced)",
}
