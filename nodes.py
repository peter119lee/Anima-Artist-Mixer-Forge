"""Anima/MiniTrainDIT cross-attention 多画师注入 v25。
===== v25 改动 =====

1. 修复 ::name::weight 显式权重没有真正关闭 normalize_weights 的问题。
   v24 文档说显式权重会尊重绝对线性强度，但实际 patch 路径仍按默认值归一化。

2. 新增 AnimaArtistPreset：
   用 balanced / strong_style / stable_seed / fast_preview / identity_guard 五个预设
   快速输出 combine/fusion/strength/advanced_options，降低新手参数门槛。

3. 新增 AnimaArtistInspector：
   在 ComfyUI UI 内直接显示画师、线性权重、有效 normalize 状态、预设参数和风险提示。

4. lowrank_avg 改为确定性的 Gram eigendecomposition 路径，避免 torch.svd_lowrank 的随机近似。

5. anchor_q cache key 加入首个 timestep/sigma，降低切换采样条件后复用旧 anchor 的风险。

===== v24 改动 =====

新增 ::name::weight 画师串语法——作用于 cross-attn 注入层的线性权重。

动机：ComfyUI 括号语法 (wlop:1.5) 作用在 CLIP 编码层，embed = f(weight)
是 text encoder 的非线性函数。weight 从 1.0 到 1.5 画风变化不是 “1.5 倍” 者什么造型变化。
v24 提供并存的另一路：画师 attention output 出来后中才乘 weight。这个乘法是纯线性，
weight=1.5 就是 weight=1.0 的 1.5 倍 delta。跨 weight 完全可预测。

语法：
  wlop                      → weight 1.0
  ::wlop::1.5               → weight 1.5 (双冗余 :: 仅美观)
  (wlop:1.1)                → 括号非线性，注入层 weight 1.0
  ::(wlop:1.1)::0.8         → 括号非线性 + 注入层 0.8 (双层叠加)

任一画师出现 ::weight 后，normalize_weights 自动失效（尊重显式输入的权重）。
带括号但未表示 ::weight 的画师仍走老逻辑。

weight 范围 [0.0, 4.0]。无效 weight（非数字）默认 1.0，不报错。

AnimaArtistPack 输出字典新增 "weights" 和 "has_explicit_weights" 字段，向后兼容老 pack
（老 pack 不带这两字段 → 默认 [1.0]*n 且 has_explicit=False）。

===== v23 改动 =====

strength 上限从 1.0 放宽到 4.0。超过 1.0 进入外推模式：
  out = base + strength × (artist - base)
其实 fusion 公式 本身在 strength>1 时 自动变为外推（数学不变），仅需：
  - UI 上限 1.0 → 4.0 （公式不动）
  - 4 处 short-circuit `>= 1.0` → `== 1.0`，避免 strength=2 时跳过外推返回 artist_total

语义：之前用户要 “画风更浓” 只能关 normalize_weights（量级 ×N 倍数、与画师数耦合、N≥4 被抦截）。
现在 strength 接管 “画风浓度” 语义，与 normalize_weights 解耦。
  - N 画师 strength=2.0 ≈ N 画师 normalize=False（但 strength 不被 N 限制）
  - normalize=False 仍保留（后兼容老工作流）
  - strength > 1 时 log info 提示进入外推模式
推荐范围 strength = 1.5-2.5。>3 过饱和。

===== v22 改动 =====
路 2 高级调优，三个可选参数（默认值 = v21 行为，完全向后兼容）：

1. anchor_seeds_count (Q1, 1-4, 默认 1)
   多个 fixed seed 跑 anchor 后 hidden state 取平均，减弱单 seed 系统偏置。
   预跑时间 × N (首次生成多 N 秒)，后续采样命中缓存 0 开销。

2. anchor_user_blend (Q4, 0.0-1.0, 默认 0.0)
   Q = blend * user_x + (1-blend) * anchor_x。
   0.0 = 纯 anchor (v21)；1.0 = 纯 user x (等价关闭 anchor_q)。
   可调跨 seed 稳定性 vs 笔触贴合度的折衰。

3. anchor_deep_layer_threshold (Q5, -1 或 可用层数, 默认 -1)
   浅层用 anchor (稳画风)、深层用 user x (贴笔触)。
   -1 表示全层都用 anchor (v21 默认)；N>=0 表示层 idx>=N 切回 user x。

===== v21 改动 =====
路 2：固定 anchor noise 替代 user-seed hidden state（artist_anchor_q）。

动机：v18~v20 测出跨 seed 偏移仍肉眼可见。根因诊断：cross-attn Q 来自
user-seed 推动的 base hidden state，无论 H' 怎么平均都脱不掉 user seed。

方案：用 fixed seed (=42) 的初始噪声跑一次完整 model forward，捕获每层
cross-attn 的输入 x（hidden state）当作 anchor。后续 user 主采样时：
- 画师 cross-attn 的 Q 用 anchor_x（完全脱钩 user seed）
- base cross-attn 的 Q 仍用 user x（保留 base 内容多样性）
- fusion 阶段把两者结合

触发：懒触发，第一次 wrapper.forward 被调用时检测 anchor cache 是否命中。
缓存 key：(x.shape, base_context_id, sigma_init)；分辨率/prompt/采样参数变
会重算。同 prompt 跨多 seed 直接命中（这是路 2 的核心收益）。

开销：anchor 预跑 = 1 完整 model forward ≈ 1 step 时间。第一次约 1 秒，
后续命中缓存 0 开销。配合 v18 static capture 时，画师 attn 仍只算前 K 步。

限制 / 风险：
- anchor x 和 user x 不在同一统计分布（fixed-seed vs user-seed），attention
  map 可能让画师贡献「位置略有偏差」（画风对、笔触位置感弱）
- 单一 anchor seed 可能引入系统性画风偏置（先跑单 seed，有症状再加多 seed）
- 与 static_capture 互斥（开 anchor_q 时忽略 static + warn）
- 与 fusion=concat_with_base 不兼容（concat 路径无独立 base/artist 计算）

===== v20 改动 =====
H' 的 K 从硬编码 3 升级为 UI 可调参数 static_capture_k（默认 6，范围 1-12）。
其他逻辑不变。
为什么默认从 3 提升到 6：
v19 (K=3) 实测画风偏移仍肉眼可见。增大 K → 前期 hidden state 多样性更大，
平均后 seed-specific 高频细节压制更彻底。K=6 是性能/效果折衷点。
更激进的用户可以拉到 8-12。
K=1 时退化为 v18（单点缓存）；K>=总 step 数时全程累加不冻结（性能略差但能跑）

===== v17 改动 =====


加回 base_preserve fusion_mode（与 EMA、lowrank_avg 兼容叠加）。

历史遗留 fusion mode（residual_shift / adain / gated / style_bias）
仍处于砍除状态。base_preserve 单独保留，因实测画风稳定。

===== 三个稳定化手段 =====

1. artist_ema_alpha (选项1)
   对每层 artist_total 做跨 step 指数滑动平均。
   仅 fusion ∈ {interpolate, base_preserve} 生效。
   alpha=0.0 关闭（默认）。新一次采样（sigma 上升）自动重置缓存。

2. combine_mode=lowrank_avg + lowrank_k (选项3，LoRA 式注入)
   N 个画师 delta 堆成矩阵，确定性投影到 top-k 主方向。
   k 越小越稳定，k=1 默认。N=1 自动 fallback 到 output_avg。

3. fusion_mode=base_preserve
   delta = artist_total - base_out
   delta_perp = delta - proj_to_base(delta)   # 每 token 独立投影
   out = base_out + strength × delta_perp
   保留 base 方向不被画师扰动，画师只能从「侧面」加偏移。

===== 兼容矩阵 =====

combine_mode × fusion_mode：
              interpolate  concat_with_base  base_preserve
output_avg        ✓             ✓                ✓
lowrank_avg       ✓             ✓                ✓
concat            ✓             ✓                ✓ (artist_total = combined attn)

EMA × fusion_mode：interpolate ✓   concat_with_base ✗   base_preserve ✓
"""

import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

FUSION_INTERPOLATE = "interpolate"
FUSION_CONCAT_WITH_BASE = "concat_with_base"
FUSION_BASE_PRESERVE = "base_preserve"

COMBINE_CONCAT = "concat"
COMBINE_OUTPUT_AVG = "output_avg"
COMBINE_LOWRANK_AVG = "lowrank_avg"

MAX_ARTISTS = 32


_STATIC_CAPTURE_K_DEFAULT = 6   # H' 跨 step 时间平均的累加步数（默认）
_STATIC_CAPTURE_K_MAX = 12      # UI 上限

_ANCHOR_SEED = 42                          # 单 anchor 默认 seed（与下面 _ANCHOR_SEEDS_POOL[0] 对齐）
_ANCHOR_SEEDS_POOL = [42, 100, 200, 300]   # 多 anchor 平均时依次取前 N 个 seed
_ANCHOR_SEEDS_MAX = 4                      # UI 上限 = len(_ANCHOR_SEEDS_POOL)
_ANCHOR_LAYER_THRESHOLD_DISABLED = -1      # Q5: -1 表示全部层都用 anchor (v21 默认)

PRESET_BALANCED = "balanced"
PRESET_STRONG_STYLE = "strong_style"
PRESET_STABLE_SEED = "stable_seed"
PRESET_FAST_PREVIEW = "fast_preview"
PRESET_IDENTITY_GUARD = "identity_guard"

PRESET_CHOICES = [
    PRESET_BALANCED,
    PRESET_STRONG_STYLE,
    PRESET_STABLE_SEED,
    PRESET_FAST_PREVIEW,
    PRESET_IDENTITY_GUARD,
]

LAYER_MODE_AUTO = "auto"
LAYER_MODE_ALL = "all_layers"
LAYER_MODE_STYLE_CORE = "style_core"
LAYER_MODE_DETAIL = "detail_layers"
LAYER_MODE_CUSTOM = "custom"

LAYER_MODE_CHOICES = [
    LAYER_MODE_AUTO,
    LAYER_MODE_ALL,
    LAYER_MODE_STYLE_CORE,
    LAYER_MODE_DETAIL,
    LAYER_MODE_CUSTOM,
]


def _base_advanced_options():
    return {
        "start_block": 0,
        "end_block": -1,
        "start_percent": 0.0,
        "end_percent": 1.0,
        "normalize_weights": True,
        "artist_ema_alpha": 0.0,
        "lowrank_k": 1,
        "artist_static_capture": False,
        "static_capture_k": _STATIC_CAPTURE_K_DEFAULT,
        "artist_anchor_q": False,
        "anchor_seeds_count": 1,
        "anchor_user_blend": 0.0,
        "anchor_deep_layer_threshold": _ANCHOR_LAYER_THRESHOLD_DISABLED,
        "layer_filter": "",
    }


def _layer_filter_for_mode(layer_mode, custom_layer_filter):
    if layer_mode == LAYER_MODE_ALL:
        return ""
    if layer_mode == LAYER_MODE_STYLE_CORE:
        return "0-18"
    if layer_mode == LAYER_MODE_DETAIL:
        return "12-63"
    if layer_mode == LAYER_MODE_CUSTOM:
        return str(custom_layer_filter or "").strip()
    return ""


def _clamp_float(value, lo, hi):
    return max(lo, min(hi, float(value)))


def _build_preset_payload(preset_name, intensity=1.0, layer_mode=LAYER_MODE_AUTO,
                          custom_layer_filter="", normalize_weights=True):
    preset_name = preset_name if preset_name in PRESET_CHOICES else PRESET_BALANCED
    intensity = _clamp_float(intensity, 0.0, 2.0)
    adv = _base_advanced_options()
    adv["normalize_weights"] = bool(normalize_weights)
    adv["layer_filter"] = _layer_filter_for_mode(layer_mode, custom_layer_filter)

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

    if preset_name != PRESET_FAST_PREVIEW:
        payload["strength"] = _clamp_float(payload["strength"] * intensity, 0.0, 4.0)
    payload["intensity"] = intensity
    payload["layer_mode"] = layer_mode
    return payload


def _merge_runtime_options(combine_mode, fusion_mode, strength,
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
    return combine_mode, fusion_mode, float(strength), adv, preset_name


def _lowrank_rows_deterministic(d_mat, k):
    """对行向量做确定性 top-k 低秩重建：D_k = U_k U_k^T D。"""
    n = int(d_mat.shape[0])
    if k >= n:
        return d_mat
    work = d_mat.to(torch.float32)
    gram = work @ work.transpose(0, 1)
    eigvals, eigvecs = torch.linalg.eigh(gram)
    order = torch.argsort(eigvals, descending=True)
    basis = eigvecs[:, order[:k]]
    return basis @ (basis.transpose(0, 1) @ work)


def _format_bool(value):
    return "on" if bool(value) else "off"


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
    """切分画师串。返回 list[str]（不解析权重，权重交给 _parse_artist_weights）。"""
    if not chain:
        return []
    s = str(chain).replace("，", ",").replace("\n", ",").replace("\r", ",")
    parts = [p.strip() for p in s.split(",")]
    return [p for p in parts if p]


def _parse_artist_weights(parts):
    """v24 新语法解析：从切好的画师串里提取 ::name::weight 的权重。

    输入: list[str]（每项可能是 'wlop'、'::wlop::1.5'、'(wlop:1.1)'、
          '(wlop:1.1):0.8'、'::(wlop:1.1)::0.8' 等）。

    返回: (names, weights, has_explicit)
      names: list[str] 给 CLIP 编码用（剥离 ::weight 后缀，括号原样保留）
      weights: list[float] 每个画师的注入权重（默认 1.0）
      has_explicit: bool 是否至少一个画师指定了 ::weight

    支持格式:
      'wlop'              → ('wlop', 1.0, False)
      '::wlop::1.5'       → ('wlop', 1.5, True)
      '(wlop:1.1)'        → ('(wlop:1.1)', 1.0, False)  # 走 CLIP 非线性，不剥离括号
      '(wlop:1.1)::0.8'   → ('(wlop:1.1)', 0.8, True)   # 双层叠加
      '::(wlop:1.1)::0.8' → ('(wlop:1.1)', 0.8, True)   # 等价写法

    无效权重（非数字）回退为默认 1.0，不报错。
    weight 卡在 [0.0, 4.0] 范围（与 strength 一致）。
    """
    names = []
    weights = []
    has_explicit = False
    for raw in parts:
        s = str(raw).strip()
        if not s:
            continue
        # 形态 1: 'name::weight' 或 '::name::weight'（前缀 :: 是可选的，仅美观）
        weight = 1.0
        explicit = False
        if "::" in s:
            head = s
            if head.startswith("::"):
                head = head[2:]
            if "::" in head:
                name_part, _, w_part = head.rpartition("::")
                w_part = w_part.strip()
                try:
                    w_val = float(w_part)
                    weight = max(0.0, min(4.0, w_val))
                    explicit = True
                    s = name_part.strip()
                except ValueError:
                    # 权重解析失败 → 当作普通文本，保留原 ::（让用户察觉）
                    pass
        if not s:
            continue
        names.append(s)
        weights.append(weight)
        if explicit:
            has_explicit = True
    return names, weights, has_explicit


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


def _project_perpendicular(delta, base):
    """剥离 delta 沿 base 方向的平行分量，返回垂直分量。

    每 token 独立投影（按最后一维 D 做内积）。
    delta_perp = delta - (delta · base_unit) * base_unit
    """
    base_norm_sq = (base * base).sum(dim=-1, keepdim=True).clamp(min=1e-8)
    proj_coef = (delta * base).sum(dim=-1, keepdim=True) / base_norm_sq
    return delta - proj_coef * base



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

    def _maybe_reset_ema(self):
        """检测 sigma 上升 → 新一次采样开始 → 重置 EMA 缓存。"""
        st = self._st
        cur = st.get("current_sigma")
        if cur is None:
            return
        prev = st.get("_ema_last_sigma")
        if prev is None or cur > prev + 1e-3:
            st["_ema_cache"] = {}
        st["_ema_last_sigma"] = cur

    def _apply_ema(self, artist_total, fusion_mode):
        """跨 step EMA 平滑（fusion ∈ {interpolate, base_preserve} 生效）。

        concat_with_base 路径不经过 artist_total，无意义。
        static_capture=True 时画师 output 已静态，EMA 无对象。
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

    def _maybe_reset_static(self):
        """检测新一次采样开始（sigma 跳升）→ 重置 static 缓存。

        单次采样内 sigma 单调下降，永不触发。
        跨次采样首步 sigma 从最大值起跳 → cur > max + EPS → 触发重置。
        CFG 双 forward / 同 sigma 重复调用，cur 不变，永不触发。
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
        """H' 跨 step 时间平均：前 K 步每步累加，第 K 步起冻结使用。

        累加用 fp32 防精度丢失，返回保持原 dtype。
        缓存 fingerprint = (x.shape, n)；分辨率切换或画师数变化自动失效。
        sigma 跳升（新一次采样）触发整体重置。
        同 sigma 重复调用（CFG 双 forward）不重复累加。
        """
        st = self._st
        if not st.get("artist_static_capture", False):
            return self._collect_artist_outputs(
                x, context, rope_emb, t_opts, individuals, fusion_mode
            )
        # static_capture 不支持 concat_with_base（x 每步变，画师 attn 含 base context 也每步变）
        if fusion_mode == FUSION_CONCAT_WITH_BASE:
            return self._collect_artist_outputs(
                x, context, rope_emb, t_opts, individuals, fusion_mode
            )

        self._maybe_reset_static()
        cache = st.setdefault("_static_cache", {})
        n = len(individuals)
        fp = (tuple(x.shape), n)

        # sigma 量化为 step 标识，None 时 fallback 用调用计数（不应该发生但稳健）
        cur_sigma = st.get("current_sigma")
        sigma_key = round(float(cur_sigma), 4) if cur_sigma is not None else None

        entry = cache.get(self._idx)
        # fingerprint 失效（分辨率变 / 画师数变）→ 该层 entry 重建
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

        # 已冻结 → 直接返回平均值
        if entry["frozen"]:
            return entry["frozen_outputs"]

        # 同一 sigma 重复调用（CFG 二次 forward 等）→ 返回当前累加平均，不重算不累加
        if sigma_key is not None and sigma_key in entry["seen_sigmas"]:
            if entry["accumulator"] is not None and entry["count"] > 0:
                inv = 1.0 / entry["count"]
                # 用 outs[0] 的 dtype 不行（这里没有 outs），从 individual 推不准；
                # 直接用 model 已知 dtype：从 context 拿
                out_dtype = context.dtype
                return [(a * inv).to(out_dtype) for a in entry["accumulator"]]
            # 累加器为空（理论上不该到这里）→ fallback 重新算一次
            return self._collect_artist_outputs(
                x, context, rope_emb, t_opts, individuals, fusion_mode
            )

        # 新 sigma → 算一次 + 累加（fp32 累加防精度丢失）
        outs = self._collect_artist_outputs(
            x, context, rope_emb, t_opts, individuals, fusion_mode
        )
        out_dtype = outs[0].dtype
        if entry["accumulator"] is None:
            entry["accumulator"] = [o.detach().to(torch.float32) for o in outs]
        else:
            for i, o in enumerate(outs):
                entry["accumulator"][i] = entry["accumulator"][i] + o.detach().to(torch.float32)
        entry["count"] += 1
        if sigma_key is not None:
            entry["seen_sigmas"].add(sigma_key)

        # 累加 K 次 → 冻结（K 从 state 读，运行时可变）
        capture_k = int(self._st.get("static_capture_k", _STATIC_CAPTURE_K_DEFAULT))
        if entry["count"] >= capture_k:

            inv = 1.0 / entry["count"]
            entry["frozen_outputs"] = [(a * inv).to(out_dtype) for a in entry["accumulator"]]
            entry["frozen"] = True
            entry["accumulator"] = None  # 释放内存
            entry["seen_sigmas"] = None
            return entry["frozen_outputs"]

        # 还没冻结 → 返回当前累加平均（用于 fusion）
        inv = 1.0 / entry["count"]
        return [(a * inv).to(out_dtype) for a in entry["accumulator"]]


    def _apply_fusion(self, base_out, artist_total, mask, fusion_mode, strength):
        """统一的 fusion 出口（处理 interpolate 和 base_preserve 两种）。

        concat_with_base 不走这里（在 _fwd_with_combined 单独处理）。
        """
        if fusion_mode == FUSION_BASE_PRESERVE:
            delta = artist_total - base_out
            delta_perp = _project_perpendicular(delta, base_out)
            out = base_out.clone()
            for i, hit in enumerate(mask):
                if hit:
                    out[i] = base_out[i] + strength * delta_perp[i]
            return out

        # 默认 interpolate
        out = base_out.clone()
        for i, hit in enumerate(mask):
            if hit:
                out[i] = base_out[i] * (1.0 - strength) + artist_total[i] * strength
        return out


    def forward(self, x, context=None, rope_emb=None, transformer_options={}):

        st = self._st

        # 路 2 anchor 预跑期间：capture 当前输入 x 到 anchor 缓存，走原始 cross_attn
        if st.get("_in_anchor_run", False):
            cache = st.setdefault("_anchor_cache", {})
            cache[self._idx] = x.detach().clone()
            return self.original(x, context, rope_emb=rope_emb,
                                 transformer_options=transformer_options)

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

        # n=1 时 lowrank_avg 没有任何意义（没有多画师方向可投影），降级到 output_avg
        if combine_mode == COMBINE_LOWRANK_AVG and len(individuals) >= 2:
            return self._fwd_lowrank_avg(
                x, context, rope_emb, transformer_options,
                individuals, weights, mask, fusion_mode, strength,
            )

        if combine_mode == COMBINE_OUTPUT_AVG or combine_mode == COMBINE_LOWRANK_AVG:
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
        static_capture = self._st.get("artist_static_capture", False)
        # static_capture 路径强制走 collect_outputs（拿 N 个独立 output 才能缓存）
        # 对 concat_with_base，static 不生效（fingerprint 失败），自动 fallback 到普通 batched 路径
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
                # 路 2：anchor_q 模式下 Q 用 anchor_x
                q_x = self._get_anchor_q_x(x)
                artist_total = self._batched_artists_forward(
                    q_x, context, rope_emb, t_opts, individuals, ws, fusion_mode
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
            q_x = self._get_anchor_q_x(x)
            for artist_i, w in zip(individuals, ws):
                artist_b = _broadcast_batch(artist_i, bsz).to(
                    device=context.device, dtype=context.dtype)
                kv = torch.cat([context, artist_b], dim=1) \
                    if fusion_mode == FUSION_CONCAT_WITH_BASE else artist_b
                out_i = self.original(q_x, kv, rope_emb=rope_emb, transformer_options=t_opts)
                artist_total = out_i * w if artist_total is None else artist_total + out_i * w


        artist_total = self._apply_ema(artist_total, fusion_mode)

        # base_preserve 必须算 base_out 才能做投影；strength == 1.0 时 interpolate 可省略。
        # strength > 1.0 外推模式仍需算 base_out（外推公式以 base 为起点）。
        if fusion_mode == FUSION_INTERPOLATE and strength == 1.0 and all(mask):
            return artist_total
        base_out = self.original(x, context, rope_emb=rope_emb, transformer_options=t_opts)
        return self._apply_fusion(base_out, artist_total, mask, fusion_mode, strength)


    def _get_anchor_q_x(self, x):
        """路 2：返回本层 cross-attn 画师 forward 使用的 Q 源。

        返回值取决于：
        - artist_anchor_q=False / 预跑失败 / 缓存未命中 → 原始 x
        - Q5: 当前 layer >= anchor_deep_layer_threshold (且 threshold >= 0) → 原始 x
        - Q4: anchor_user_blend > 0 → blend * x + (1-blend) * anchor_x
        - 其他 → anchor_x (v21 默认)

        shape 不匹配（如 batch 变化）会 fallback 到 user x。
        """
        st = self._st
        if not st.get("artist_anchor_q", False):
            return x
        if st.get("_anchor_failed", False):
            return x

        # Q5: 深层切回 user x
        threshold = int(st.get("anchor_deep_layer_threshold", _ANCHOR_LAYER_THRESHOLD_DISABLED))
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

        # Q4: 与 user x 加权融合
        blend = float(st.get("anchor_user_blend", 0.0))
        blend = max(0.0, min(1.0, blend))
        if blend > 0.0:
            return blend * x + (1.0 - blend) * anchor_x
        return anchor_x

    def _collect_artist_outputs(self, x, context, rope_emb, t_opts,
                                individuals, fusion_mode):
        """算 N 个画师各自的 attention output。返回 list of (B, T, D)。

        路 2：artist_anchor_q 开启时 Q 来源从 x 替换为 anchor_x。
        """
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
                        "[AnimaCrossAttn] batched outputs 失败，回退串行: %s", e,
                    )
                    self._st["_warned_batched"] = True
                    self._st["_disable_batched"] = True
        outs = []
        for artist_i in individuals:
            artist_b = _broadcast_batch(artist_i, bsz).to(
                device=context.device, dtype=context.dtype)
            kv = torch.cat([context, artist_b], dim=1) \
                if fusion_mode == FUSION_CONCAT_WITH_BASE else artist_b
            out_i = self.original(q_x, kv, rope_emb=rope_emb, transformer_options=t_opts)
            outs.append(out_i)
        return outs

    def _batched_artists_outputs_only(self, x, context, rope_emb, t_opts,
                                       individuals, fusion_mode):
        """N 个画师 forward batch 化，返回 list of (B, T, D)（不做加权和）。"""
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
        return [out[i] for i in range(n)]

    def _fwd_lowrank_avg(self, x, context, rope_emb, t_opts,
                         individuals, weights, mask, fusion_mode, strength):
        """LoRA 式低秩注入：N 个画师 delta 投影到 top-k 主方向子空间后加权融合。

        delta_i = A_i - A_base
        D = stack(delta_i)              # (N, M)
        D_lowrank = topk_rowspace_project(D, k)
        delta_avg = sum(w_i * D_lowrank[i])
        artist_total = A_base + delta_avg
        """
        if self._st.get("normalize_weights", True):
            ws = _normalize_weights(weights)
        else:
            ws = list(weights)
        n = len(individuals)
        k = int(self._st.get("lowrank_k", 1))
        k = max(1, min(k, n))

        # 1. 算 N 个画师 attn output（static_capture 命中则用缓存）
        artist_outs = self._get_artist_outputs_with_cache(
            x, context, rope_emb, t_opts, individuals, fusion_mode
        )


        # 2. 算 base attn output（lowrank 模式必须算，无论 strength）
        base_out = self.original(x, context, rope_emb=rope_emb, transformer_options=t_opts)
        out_dtype = base_out.dtype

        # 3. 堆 delta，低秩投影全程 fp32
        A = torch.stack(artist_outs, dim=0).to(torch.float32)   # (N, B, T, D)
        base_f32 = base_out.to(torch.float32).unsqueeze(0)      # (1, B, T, D)
        delta = A - base_f32                                    # (N, B, T, D)

        orig_shape = delta.shape
        D_mat = delta.reshape(n, -1)                            # (N, M)

        if k < n:
            try:
                D_lowrank = _lowrank_rows_deterministic(D_mat, k)
            except Exception as e:
                if not self._st.get("_warned_svd", False):
                    logger.warning(
                        "[AnimaCrossAttn] L%d lowrank_avg 失败，本步退化为 output_avg: %s",
                        self._idx, e,
                    )
                    self._st["_warned_svd"] = True
                D_lowrank = D_mat
        else:
            # k >= n 数学上等价于 output_avg（不投影）
            D_lowrank = D_mat

        # 4. 沿 N 维加权和
        w_t = torch.tensor(ws, device=D_lowrank.device, dtype=D_lowrank.dtype).view(n, 1)
        delta_avg = (D_lowrank * w_t).sum(dim=0)                # (M,)
        delta_avg = delta_avg.reshape(orig_shape[1:]).to(out_dtype)  # (B, T, D)

        artist_total = base_out + delta_avg

        # 5. EMA（如果开了）
        artist_total = self._apply_ema(artist_total, fusion_mode)

        # 6. fusion
        if fusion_mode == FUSION_INTERPOLATE and strength == 1.0 and all(mask):
            return artist_total
        return self._apply_fusion(base_out, artist_total, mask, fusion_mode, strength)


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

        if fusion_mode in (FUSION_INTERPOLATE, FUSION_BASE_PRESERVE):
            base_out = self.original(x, context, rope_emb=rope_emb, transformer_options=t_opts)
            # 路 2：anchor_q 模式下画师 attn 的 Q 用 anchor_x
            q_x = self._get_anchor_q_x(x)
            # combined attn output 也走 static 缓存（n=1 视角的列表语义）
            static_capture = self._st.get("artist_static_capture", False)
            if static_capture:
                self._maybe_reset_static()
                cache = self._st.setdefault("_static_cache", {})
                cached = cache.get(self._idx)
                fp = (tuple(x.shape), -1)  # -1 标记 combined 路径
                if cached is not None and cached.get("_fp") == fp:
                    artist_out = cached["outputs"][0]
                else:
                    artist_out = self.original(q_x, artist_b, rope_emb=rope_emb, transformer_options=t_opts)
                    cache[self._idx] = {"outputs": [artist_out.detach()], "_fp": fp}
            else:
                artist_out = self.original(q_x, artist_b, rope_emb=rope_emb, transformer_options=t_opts)
            artist_out = self._apply_ema(artist_out, fusion_mode)

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



def _make_sigma_capture(state, prev_wrapper):
    """包装 model forward：
    1. 捕获当前 sigma。
    2. 检测新一次采样（sigma 跳升） → 清空跨采样缓存。
    3. anchor_q 开启且缓存未命中 → 预跑一次 anchor forward 填充缓存。
    4. 执行真正的 user forward。
    """
    def wrapper(apply_model, options):
        ts = options.get("timestep")
        cur_sigma = None
        if ts is not None:
            try:
                cur_sigma = float(ts.flatten()[0].item())
                state["current_sigma"] = cur_sigma
            except Exception:
                pass

        # anchor 缓存不随 sigma 跳升失效——同 prompt 跨多 seed 的 fingerprint 一致，走命中分支。
        # 只有 fingerprint (x.shape, id(base_context), first_timestep) 变化才重跑 anchor。

        # anchor_q 开启 → 检查缓存是否需要预跑
        if state.get("artist_anchor_q", False) and not state.get("_anchor_failed", False):
            user_x = options.get("input")
            user_ts = options.get("timestep")
            c_dict = options.get("c", {}) or {}
            if user_x is not None and user_ts is not None and c_dict:
                _maybe_run_anchor(state, user_x, user_ts, c_dict)

        if prev_wrapper is not None:
            return prev_wrapper(apply_model, options)
        return apply_model(options["input"], options["timestep"], **options["c"])
    return wrapper


def _maybe_run_anchor(state, user_x, user_timestep, c_dict):
    """路 2 anchor 预跑：如果缓存未命中，用 fixed seed 噪声跑一次完整 model forward，
    捕获每层 cross-attn 输入 x 到 state["_anchor_cache"][layer_idx]。

    预跑期间 state["_in_anchor_run"]=True，wrapper.forward 走 capture 分支：
    记录 x 后调用原始 cross_attn（不做画师注入）。

    调用位置在 model_function_wrapper 里，此时主 model forward 尚未开始，不会递归。

    c_dict 包含 context (raw t5 embedding) 以及 t5xxl_ids / t5xxl_weights（可选），
    调 dm.__call__() 走 Anima.forward 完整路径（内部会调 preprocess_text_embeds）。
    """
    base_context = c_dict.get("context")
    if base_context is None:
        return

    # CFG 下取 cond 那行作为 anchor 词条
    transformer_options = c_dict.get("transformer_options", {}) or {}
    if base_context.dim() >= 2 and base_context.shape[0] > 1:
        cou = transformer_options.get("cond_or_uncond")
        if cou is not None and 0 in cou:
            cond_idx = cou.index(0)
            base_context = base_context[cond_idx:cond_idx + 1]
        else:
            base_context = base_context[:1]

    cache_key = state.get("_anchor_cache_key")
    try:
        sigma_key = round(float(user_timestep.flatten()[0].item()), 4)
    except Exception:
        sigma_key = None
    new_key = (tuple(user_x.shape), id(c_dict.get("context")), sigma_key)
    if cache_key == new_key and state.get("_anchor_cache"):
        return  # 命中缓存，不重跑

    dm = state["dm_ref"]

    # 打开 capture 开关（多 seed 循环里会在每轮开始清空缓存）
    state["_anchor_cache"] = {}
    state["_in_anchor_run"] = True

    # base_context batch 对齐到 user_x.shape[0]
    bsz = user_x.shape[0]
    if base_context.shape[0] != bsz:
        if base_context.shape[0] == 1:
            ctx_for_anchor = base_context.expand(bsz, -1, -1)
        else:
            ctx_for_anchor = base_context[:1].expand(bsz, -1, -1)
    else:
        ctx_for_anchor = base_context
    ctx_for_anchor = ctx_for_anchor.contiguous().to(device=user_x.device, dtype=user_x.dtype)

    # 准备 t5xxl_ids / t5xxl_weights（如果主路径传了）同样对齐 batch
    anchor_kwargs = {}
    for key in ("t5xxl_ids", "t5xxl_weights"):
        v = c_dict.get(key)
        if v is None or not torch.is_tensor(v):
            continue
        if v.shape[0] != bsz:
            if v.shape[0] == 1:
                v = v.expand(bsz, *v.shape[1:])
            else:
                v = v[:1].expand(bsz, *v.shape[1:])
        anchor_kwargs[key] = v.contiguous()

    # 隔离 transformer_options：不带 cond_or_uncond / patches
    safe_opts = dict(transformer_options) if isinstance(transformer_options, dict) else {}
    safe_opts.pop("cond_or_uncond", None)
    safe_opts.pop("patches", None)
    anchor_kwargs["transformer_options"] = safe_opts

    try:
        with torch.no_grad():
            # 手动调 preprocess_text_embeds（如果主路径传了 t5xxl_ids），然后走 dm._forward。
            # 这样跳过所有 wrappers，避免递归。
            t5xxl_ids = anchor_kwargs.pop("t5xxl_ids", None)
            t5xxl_weights = anchor_kwargs.pop("t5xxl_weights", None)
            if t5xxl_ids is not None and hasattr(dm, "preprocess_text_embeds"):
                processed_ctx = dm.preprocess_text_embeds(
                    ctx_for_anchor, t5xxl_ids, t5xxl_weights=t5xxl_weights,
                )
            else:
                processed_ctx = ctx_for_anchor
            t_opts_for_anchor = anchor_kwargs.get("transformer_options", {})

            # Q1: 多 anchor seed 平均。anchor_seeds_count 控制跑几个 seed (1=单 seed v21)
            seeds_count = max(1, min(int(state.get("anchor_seeds_count", 1)), _ANCHOR_SEEDS_MAX))
            seeds = _ANCHOR_SEEDS_POOL[:seeds_count]

            accumulator = {}   # layer_idx -> sum of hidden states (fp32)
            for seed in seeds:
                # 为每个 seed 生成 anchor 噪声
                gen = torch.Generator(device=user_x.device)
                gen.manual_seed(seed)
                anchor_x_k = torch.randn(
                    user_x.shape, generator=gen,
                    device=user_x.device, dtype=user_x.dtype,
                )
                # 跳过不可记录状态 → 每跳一次清空 _anchor_cache
                state["_anchor_cache"] = {}
                _ = dm._forward(
                    anchor_x_k, user_timestep, processed_ctx,
                    transformer_options=t_opts_for_anchor,
                )
                # 本轮捕获累加到 accumulator (fp32)
                for layer_idx, hidden in state["_anchor_cache"].items():
                    if layer_idx not in accumulator:
                        accumulator[layer_idx] = hidden.to(torch.float32)
                    else:
                        accumulator[layer_idx] = accumulator[layer_idx] + hidden.to(torch.float32)

            # 平均后转回原 dtype，写回主 _anchor_cache
            inv = 1.0 / max(1, seeds_count)
            avg_dtype = user_x.dtype
            state["_anchor_cache"] = {
                idx: (acc * inv).to(avg_dtype) for idx, acc in accumulator.items()
            }
    except Exception as e:
        logger.warning(
            "[AnimaCrossAttn] anchor 预跑失败，本次退化为 v20 行为: %s", e,
        )
        state["_anchor_cache"] = {}
        state["_anchor_failed"] = True
    finally:
        state["_in_anchor_run"] = False

    if state["_anchor_cache"]:
        state["_anchor_cache_key"] = new_key
        if not state.get("_warned_anchor_ok", False):
            logger.info(
                "[AnimaCrossAttn] anchor 预跑完成，捕获 %d 层 hidden state",
                len(state["_anchor_cache"]),
            )
            state["_warned_anchor_ok"] = True


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
                        "画师串。用英文/中文逗号或换行分隔。\n"
                        "例: wlop, sakimichan, krenz\n"
                        "\n"
                        "支持两种权重语法（可以共存不互斥）：\n"
                        "  1) 括号语法 (wlop:1.5)——作用于 CLIP 编码层，非线性\n"
                        "  2) ::weight 语法 ::wlop::1.5——作用于 cross-attn 注入层，线性\n"
                        "\n"
                        "默认 weight=1.0。范围 [0.0, 4.0]。\n"
                        "::weight 与 括号可以叠加: ::(wlop:1.1)::0.8\n"
                        "\n"
                        "任何画师指定了 ::weight 后，normalize_weights 自动失效\n"
                        "（尊重用户输入的按重）。"
                    )
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
        parts = _split_artist_chain(artist_chain)
        names, parsed_weights, has_explicit = _parse_artist_weights(parts)
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
                "weights": [],
                "has_explicit_weights": False,
                "base_prompt": base,
                "base_conditioning": base_conditioning,
            },)

        if len(names) > MAX_ARTISTS:
            logger.warning(
                "[AnimaArtistPack] 画师数 %d 超过上限 %d，截断",
                len(names), MAX_ARTISTS,
            )
            names = names[:MAX_ARTISTS]
            parsed_weights = parsed_weights[:MAX_ARTISTS]

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

        if has_explicit:
            logger.info(
                "[AnimaArtistPack] 检测到 %d 个画师指定了 ::weight 语法，将走线性注入路径",
                sum(1 for w in parsed_weights if w != 1.0),
            )

        return ({
            "conditionings": conditionings,
            "labels": names,
            "weights": parsed_weights,
            "has_explicit_weights": has_explicit,
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
                    "tooltip": (
                        "True: weights 归一化为相对比例。False: weights 直接作为独立强度\n"
                        "\n"
                        "v24 语义变化：如果 artist_chain 用了 ::weight 语法指定权重，\n"
                        "本开关自动失效（始终尊重用户显式输入的权重，不做归一化）。"
                    )
                }),
                "artist_ema_alpha": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 0.95, "step": 0.05,
                    "tooltip": (
                        "跨 step EMA 平滑系数（仅 fusion=interpolate 生效）。\n"
                        "对每层 artist_total 做指数滑动平均，缓解多画师场景下\n"
                        "跨 seed 主导画师切换导致的画风漂移。\n"
                        "0.0:   关闭（默认）\n"
                        "0.3-0.5: 轻度平滑\n"
                        "0.5-0.8: 中重度平滑\n"
                        ">0.8:  强平滑，画风可能跟不上 base 内容变化\n"
                        "新一次采样（sigma 上升）自动重置缓存。"
                    ),
                }),
                "lowrank_k": ("INT", {
                    "default": 1, "min": 1, "max": MAX_ARTISTS, "step": 1,
                    "tooltip": (
                        "LoRA 式低秩注入维度（仅 combine_mode=lowrank_avg 生效）。\n"
                        "对 N 个画师 delta 做确定性低秩投影，截断到 top-k 主方向。\n"
                        "k=1: 所有画师沿单一共识方向，跨 seed 最稳，画风最同质\n"
                        "k=2-3: 保留主要画风方向，画师间允许少量差异（推荐范围）\n"
                        "k>=N: 等价于 output_avg（不投影）\n"
                        "N=1 时自动 fallback 到 output_avg。"
                    ),
                }),
                "artist_static_capture": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "H' 跨 step 时间平均（v20）：前 K 步每步算画师 attention 并累加，\n"
                        "第 K 步起冻结，后续 step 直接复用平均值。\n"
                        "理论：x_t1、x_t2... 已被去噪过，seed-specific 细节比 x_t0 更弱，\n"
                        "前 K 步平均 ≈「降噪轨迹早期的画师响应期望」，跨 seed 偏移更小。\n"
                        "K 由 static_capture_k 控制（默认 6）。\n"
                        "兼容：output_avg / lowrank_avg + interpolate / base_preserve。\n"
                        "不兼容：fusion=concat_with_base（自动忽略）。\n"
                        "与 EMA 语义互斥（开了 static 后 EMA 自动忽略）。"
                    ),
                }),
                "static_capture_k": ("INT", {
                    "default": _STATIC_CAPTURE_K_DEFAULT,
                    "min": 1, "max": _STATIC_CAPTURE_K_MAX, "step": 1,
                    "tooltip": (
                        "H' 跨 step 时间平均的累加步数（仅 artist_static_capture=True 生效）。\n"
                        "K=1: 退化为 v18 单点缓存（最快但跨 seed 偏移最大）\n"
                        "K=3: 早期保守平均（v19 行为）\n"
                        "K=6: 推荐起点（v20 默认）\n"
                        "K=8-10: 进一步压制偏移，前 K 步每步多算 N 次画师 cross-attn\n"
                        "K=12: 上限，30 step 时几乎前半程都在累加（性能损失最大）\n"
                        "若总采样步数 < K，自动按当前累加平均工作（不会出错）。"
                    ),
                }),
                "artist_anchor_q": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "路 2（v21）：画师 cross-attn 的 Q 用 fixed-seed anchor hidden state，\n"
                        "完全脱钩 user seed。比 H' 更激进的跨 seed 稳定化手段。\n"
                        "\n"
                        "工作原理：首次采样前用 fixed seed (=42) 预跑一次完整 model forward，\n"
                        "捕获每层 cross-attn 输入 hidden state 作为 anchor。后续 user step 中画师\n"
                        "进 attn 时 Q 用 anchor_x，base 仍用 user x。\n"
                        "\n"
                        "开销：预跑 ≈ 1 step 时间（首次生成多 1 秒）；同 prompt 跨 seed 命中缓存 0 开销。\n"
                        "\n"
                        "与 static_capture 互斥（开 anchor_q 时 static 自动忽略）。\n"
                        "与 fusion=concat_with_base 不兼容（concat 路径不独立调用画师 forward）。\n"
                        "\n"
                        "风险：画风位置可能与当前图略不贴（画风对、笔触位置感弱），\n"
                        "这是路 2 的本质上限。如遇到该症状 fallback 到 v20。"
                    ),
                }),
                "anchor_seeds_count": ("INT", {
                    "default": 1, "min": 1, "max": _ANCHOR_SEEDS_MAX, "step": 1,
                    "tooltip": (
                        "Q1 (v22): anchor 预跑用的 seed 个数。仅 anchor_q=True 生效。\n"
                        "1: 单 seed (v21 默认)\n"
                        "2-4: 多个 fixed seed 跑 anchor后 hidden state 取平均，减弱单 seed 系统偏置\n"
                        "\n"
                        "运行时预跑时间 × N（首次生成多 N 秒），后续采样命中缓存 0 开销。\n"
                        "如果画风跨 seed 稳但总体偏黑/偏亮之类，提高到 2-3 可缓解。"
                    ),
                }),
                "anchor_user_blend": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": (
                        "Q4 (v22): anchor / user x 混合比例。仅 anchor_q=True 生效。\n"
                        "Q = blend * user_x + (1-blend) * anchor_x\n"
                        "\n"
                        "0.0: 纯 anchor (v21 默认、跨 seed 最稳)\n"
                        "0.3-0.5: 混合，稳定性与贴合度折衰\n"
                        "1.0: 纯 user x (等价关闭 anchor_q)\n"
                        "\n"
                        "如果跨 seed 画风可以接受但笔触位置不贴当前图，温和提高 blend。"
                    ),
                }),
                "anchor_deep_layer_threshold": ("INT", {
                    "default": _ANCHOR_LAYER_THRESHOLD_DISABLED,
                    "min": _ANCHOR_LAYER_THRESHOLD_DISABLED, "max": 64, "step": 1,
                    "tooltip": (
                        "Q5 (v22): 只在浅层用 anchor，深层切回 user x。仅 anchor_q=True 生效。\n"
                        "-1: 禁用阈值，所有层都用 anchor (v21 默认)\n"
                        "N>=0: 层 idx < N 用 anchor，idx >= N 用 user x\n"
                        "\n"
                        "逻辑：浅层 cross-attn 决定「画风方向」，深层决定「笔触贴合」。\n"
                        "示例：28 层模型设 N=14 则前半用 anchor (稳画风)、后半用 user x (贴笔触)。\n"
                        "\n"
                        "与 anchor_user_blend 可以叠加：深层切回 user x 后 blend 不再生效。"
                    ),
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
              artist_ema_alpha=0.0, lowrank_k=1, artist_static_capture=False,
              static_capture_k=_STATIC_CAPTURE_K_DEFAULT, artist_anchor_q=False,
              anchor_seeds_count=1, anchor_user_blend=0.0,
              anchor_deep_layer_threshold=_ANCHOR_LAYER_THRESHOLD_DISABLED,
              layer_filter=""):
        return ({
            "start_block": int(start_block),
            "end_block": int(end_block),
            "start_percent": float(start_percent),
            "end_percent": float(end_percent),
            "normalize_weights": bool(normalize_weights),
            "artist_ema_alpha": float(artist_ema_alpha),
            "lowrank_k": int(lowrank_k),
            "artist_static_capture": bool(artist_static_capture),
            "static_capture_k": int(static_capture_k),
            "artist_anchor_q": bool(artist_anchor_q),
            "anchor_seeds_count": int(anchor_seeds_count),
            "anchor_user_blend": float(anchor_user_blend),
            "anchor_deep_layer_threshold": int(anchor_deep_layer_threshold),
            "layer_filter": str(layer_filter or ""),
        },)


class AnimaArtistPreset:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "preset": (PRESET_CHOICES, {
                    "default": PRESET_BALANCED,
                    "tooltip": (
                        "一键工作模式。\n"
                        "balanced: 推荐默认，轻 EMA，画风稳定但不过度保守\n"
                        "strong_style: 更浓画风，strength 外推到 1.65\n"
                        "stable_seed: lowrank + static capture，优先跨 seed 稳定\n"
                        "fast_preview: concat 路径，优先速度，适合找图\n"
                        "identity_guard: base_preserve + lowrank，尽量保主 prompt 身份/构图"
                    ),
                }),
                "intensity": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "预设强度倍率。fast_preview 不使用 strength，其他预设会乘到 strength 上。",
                }),
                "normalize_weights": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "预设里的默认 normalize_weights。artist_chain 若用了 ::weight，运行时仍会自动关闭。",
                }),
                "layer_mode": (LAYER_MODE_CHOICES, {
                    "default": LAYER_MODE_AUTO,
                    "tooltip": (
                        "层范围快捷选择。\n"
                        "auto/all_layers: 全层\n"
                        "style_core: 0-18，偏整体画风\n"
                        "detail_layers: 12-63，偏细节和笔触\n"
                        "custom: 使用 custom_layer_filter"
                    ),
                }),
                "custom_layer_filter": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "layer_mode=custom 时生效。例: 0,3,5-10,-1",
                }),
            },
        }

    RETURN_TYPES = ("ANIMA_PRESET", "ANIMA_OPTS", "STRING")
    RETURN_NAMES = ("preset", "advanced_options", "summary")
    FUNCTION = "build"
    CATEGORY = "Anima/CrossAttn"

    def build(self, preset, intensity, normalize_weights, layer_mode, custom_layer_filter):
        payload = _build_preset_payload(
            preset, intensity, layer_mode, custom_layer_filter, normalize_weights,
        )
        adv = payload["advanced_options"]
        summary = "\n".join([
            f"Preset: {payload['preset']}",
            f"combine_mode: {payload['combine_mode']}",
            f"fusion_mode: {payload['fusion_mode']}",
            f"strength: {payload['strength']:.2f}",
            f"normalize_weights: {_format_bool(adv.get('normalize_weights', True))}",
            f"EMA alpha: {float(adv.get('artist_ema_alpha', 0.0)):.2f}",
            f"lowrank_k: {int(adv.get('lowrank_k', 1))}",
            f"static_capture: {_format_bool(adv.get('artist_static_capture', False))}",
            f"static_capture_k: {int(adv.get('static_capture_k', _STATIC_CAPTURE_K_DEFAULT))}",
            f"layer_filter: {adv.get('layer_filter') or 'all'}",
        ])
        return {"ui": {"text": [summary]}, "result": (payload, adv, summary)}


class AnimaArtistInspector:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "artist_pack": ("ANIMA_PACK",),
            },
            "optional": {
                "combine_mode": (
                    [COMBINE_CONCAT, COMBINE_OUTPUT_AVG, COMBINE_LOWRANK_AVG],
                    {"default": COMBINE_OUTPUT_AVG},
                ),
                "fusion_mode": (
                    [FUSION_INTERPOLATE, FUSION_CONCAT_WITH_BASE, FUSION_BASE_PRESERVE],
                    {"default": FUSION_INTERPOLATE},
                ),
                "strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 4.0, "step": 0.05,
                }),
                "advanced_options": ("ANIMA_OPTS",),
                "preset": ("ANIMA_PRESET",),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("report",)
    FUNCTION = "inspect"
    CATEGORY = "Anima/CrossAttn"
    OUTPUT_NODE = True

    def inspect(self, artist_pack, combine_mode=COMBINE_OUTPUT_AVG,
                fusion_mode=FUSION_INTERPOLATE, strength=1.0,
                advanced_options=None, preset=None):
        if not isinstance(artist_pack, dict):
            report = "Anima Artist Inspector\nERROR: artist_pack 不是有效的 ANIMA_PACK。"
            return {"ui": {"text": [report]}, "result": (report,)}

        labels = list(artist_pack.get("labels") or [])
        weights = artist_pack.get("weights")
        if not isinstance(weights, (list, tuple)) or len(weights) != len(labels):
            weights = [1.0] * len(labels)
        weights = [float(w) for w in weights]
        has_explicit = bool(artist_pack.get("has_explicit_weights", False))
        base_prompt = str(artist_pack.get("base_prompt", "") or "")

        combine_mode, fusion_mode, strength, adv, preset_name = _merge_runtime_options(
            combine_mode, fusion_mode, strength, advanced_options, preset,
        )
        requested_normalize = bool(adv.get("normalize_weights", True))
        effective_normalize = requested_normalize and not has_explicit
        weight_sum = sum(abs(w) for w in weights)

        lines = [
            "Anima Artist Mixer Inspector",
            "",
            f"preset: {preset_name or '(none)'}",
            f"artists: {len(labels)}",
            f"base_prompt: {'yes' if base_prompt else 'empty'}",
            f"combine_mode: {combine_mode}",
            f"fusion_mode: {fusion_mode}",
            f"strength: {float(strength):.2f}",
            f"requested normalize_weights: {_format_bool(requested_normalize)}",
            f"effective normalize_weights: {_format_bool(effective_normalize)}",
            f"effective linear weight sum: {weight_sum:.3f}",
            f"layer_filter: {adv.get('layer_filter') or 'all'}",
            f"sigma range percent: {float(adv.get('start_percent', 0.0)):.3f} - "
            f"{float(adv.get('end_percent', 1.0)):.3f}",
            f"EMA alpha: {float(adv.get('artist_ema_alpha', 0.0)):.2f}",
            f"lowrank_k: {int(adv.get('lowrank_k', 1))}",
            f"static_capture: {_format_bool(adv.get('artist_static_capture', False))} "
            f"(K={int(adv.get('static_capture_k', _STATIC_CAPTURE_K_DEFAULT))})",
            f"anchor_q: {_format_bool(adv.get('artist_anchor_q', False))}",
            "",
            "artists:",
        ]

        if labels:
            for idx, (label, weight) in enumerate(zip(labels, weights), start=1):
                lines.append(f"  {idx}. {label} :: {weight:.3g}")
        else:
            lines.append("  (none)")

        warnings = []
        if not labels:
            warnings.append("没有画师；CrossAttn 会原样返回 base prompt。")
        if has_explicit and requested_normalize:
            warnings.append("检测到 ::weight；运行时会自动关闭 normalize_weights，这是正确行为。")
        if not effective_normalize and weight_sum > 1.5:
            warnings.append("线性权重和 > 1.5，画风可能过浓或过曝。")
        if (
            adv.get("artist_static_capture", False)
            and adv.get("artist_anchor_q", False)
        ):
            warnings.append("static_capture 与 anchor_q 互斥；CrossAttn 会关闭 static_capture。")
        if fusion_mode == FUSION_CONCAT_WITH_BASE and adv.get("artist_anchor_q", False):
            warnings.append("concat_with_base 不支持 anchor_q；CrossAttn 会关闭 anchor_q。")
        if fusion_mode == FUSION_CONCAT_WITH_BASE and adv.get("artist_static_capture", False):
            warnings.append("concat_with_base 不支持 static_capture；会退回普通路径。")
        if combine_mode == COMBINE_LOWRANK_AVG and len(labels) <= 1:
            warnings.append("只有 1 个画师时 lowrank_avg 没意义，会自动按 output_avg 工作。")

        lines.append("")
        lines.append("warnings:")
        if warnings:
            lines.extend(f"  - {w}" for w in warnings)
        else:
            lines.append("  - no obvious configuration risk")

        report = "\n".join(lines)
        return {"ui": {"text": [report]}, "result": (report,)}






class AnimaArtistCrossAttn:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "artist_pack": ("ANIMA_PACK",),
                "combine_mode": (
                    [COMBINE_CONCAT, COMBINE_OUTPUT_AVG, COMBINE_LOWRANK_AVG],
                    {"default": COMBINE_OUTPUT_AVG},
                ),

                "fusion_mode": (
                    [FUSION_INTERPOLATE, FUSION_CONCAT_WITH_BASE, FUSION_BASE_PRESERVE],
                    {
                        "default": FUSION_INTERPOLATE,
                        "tooltip": (
                            "interpolate: out = lerp(base, artist, strength) 经典融合\n"
                            "concat_with_base: KV=[base; artist] 单次 forward，attn 自然融合\n"
                            "base_preserve: 剥离画师沿 base 方向的分量，只保留垂直偏移\n"
                            "  out = base + strength × proj_perp(artist - base)\n"
                            "  base 内容方向不被扰动，画师只能从侧面加偏移\n"
                            "  与 lowrank_avg / EMA 兼容叠加"
                        ),
                    },
                ),

                "strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 4.0, "step": 0.05,
                    "tooltip": (
                        "画师注入强度。\n"
                        "0.0-1.0: 插值模式 lerp(base, artist, strength)\n"
                        "  0.0 = 纯 base，1.0 = 纯 artist 替换\n"
                        "1.0-4.0: 外推模式 base + strength × (artist - base)\n"
                        "  画师贡献被放大，画风浓度上升。等价于关 normalize_weights\n"
                        "  的 hack 路径但更可控（与画师数解耦）。\n"
                        "  推荐 1.5-2.5，>3 容易过饱和。\n"
                        "  N 画师 strength=2 ≈ N 画师 normalize=False。"
                    ),
                }),
                "enabled": ("BOOLEAN", {"default": True}),
                "apply_to_uncond": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "advanced_options": ("ANIMA_OPTS",),
                "preset": ("ANIMA_PRESET",),
            },
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING")
    RETURN_NAMES = ("model", "base_prompt")
    FUNCTION = "patch"
    CATEGORY = "Anima/CrossAttn"

    def patch(self, model, artist_pack, combine_mode, fusion_mode,
              strength, enabled, apply_to_uncond, advanced_options=None, preset=None):
        combine_mode, fusion_mode, strength, adv, preset_name = _merge_runtime_options(
            combine_mode, fusion_mode, strength, advanced_options, preset,
        )
        sb = int(adv.get("start_block", 0))
        eb = int(adv.get("end_block", -1))
        start_percent = float(adv.get("start_percent", 0.0))
        end_percent = float(adv.get("end_percent", 1.0))
        normalize_w = bool(adv.get("normalize_weights", True))
        artist_ema_alpha = float(adv.get("artist_ema_alpha", 0.0))
        lowrank_k = int(adv.get("lowrank_k", 1))
        artist_static_capture = bool(adv.get("artist_static_capture", False))
        static_capture_k = int(adv.get("static_capture_k", _STATIC_CAPTURE_K_DEFAULT))
        static_capture_k = max(1, min(static_capture_k, _STATIC_CAPTURE_K_MAX))
        artist_anchor_q = bool(adv.get("artist_anchor_q", False))
        anchor_seeds_count = int(adv.get("anchor_seeds_count", 1))
        anchor_seeds_count = max(1, min(anchor_seeds_count, _ANCHOR_SEEDS_MAX))
        anchor_user_blend = float(adv.get("anchor_user_blend", 0.0))
        anchor_user_blend = max(0.0, min(1.0, anchor_user_blend))
        anchor_deep_layer_threshold = int(
            adv.get("anchor_deep_layer_threshold", _ANCHOR_LAYER_THRESHOLD_DISABLED)
        )
        layer_filter_text = str(adv.get("layer_filter", "") or "")

        use_sigma_range = (start_percent > 0.0) or (end_percent < 1.0)
        # EMA / static_capture / anchor_q 都需要 sigma capture 检测新一次采样
        need_sigma_capture = (
            use_sigma_range or (artist_ema_alpha > 0.0)
            or artist_static_capture or artist_anchor_q
        )

        # 互斥校验
        if artist_static_capture and artist_ema_alpha > 0.0:
            logger.info(
                "[AnimaCrossAttn] artist_static_capture=True 时 artist_ema_alpha=%.2f 自动忽略"
                "（画师 output 已静态，EMA 无对象）。",
                artist_ema_alpha,
            )
        if artist_static_capture and fusion_mode == FUSION_CONCAT_WITH_BASE:
            logger.warning(
                "[AnimaCrossAttn] artist_static_capture=True 但 fusion=concat_with_base 不兼容"
                "（x 每步变，画师 attn 输出不可缓存）。本次 static 自动忽略。"
            )
        if artist_anchor_q and artist_static_capture:
            logger.warning(
                "[AnimaCrossAttn] artist_anchor_q=True 与 artist_static_capture=True 互斥，"
                "本次 static_capture 自动关闭（anchor_q 优先）。"
            )
            artist_static_capture = False
        if artist_anchor_q and fusion_mode == FUSION_CONCAT_WITH_BASE:
            logger.warning(
                "[AnimaCrossAttn] artist_anchor_q=True 与 fusion=concat_with_base 不兼容。"
                "本次 anchor_q 自动关闭。"
            )
            artist_anchor_q = False



        # base_preserve 与 concat_with_base 互斥：前者需要分别拿 base_out 和 artist_total，
        # 后者把它们拼成单次 forward 无法分离
        if fusion_mode == FUSION_BASE_PRESERVE and combine_mode == COMBINE_CONCAT:
            # concat 模式 + base_preserve 是允许的（combined attn 当 artist_total）
            pass

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
        # v24: 从 pack 拿画师权重（默认 [1.0]*n，向后兼容老 pack）
        parsed_weights = artist_pack.get("weights")
        has_explicit_weights = bool(artist_pack.get("has_explicit_weights", False))
        if isinstance(parsed_weights, (list, tuple)) and len(parsed_weights) == n:
            user_weights = [float(w) for w in parsed_weights]
        else:
            user_weights = [1.0] * n
            has_explicit_weights = False

        if has_explicit_weights and normalize_w:
            normalize_w = False
            logger.info(
                "[AnimaCrossAttn] 检测到 ::weight 显式线性权重，normalize_weights 自动关闭。"
            )

        # base_preserve 在 strength 很小（<0.3）时几乎无效（垂直分量 × 小系数 = 微小偏移），
        # 在 strength=1.0 时也仍然是「不强制对齐 artist」；这是设计目标。
        # 这里只做提示，不做强限制。
        if fusion_mode == FUSION_BASE_PRESERVE and float(strength) < 0.3:
            logger.info(
                "[AnimaCrossAttn] fusion=base_preserve 在 strength=%.2f (<0.3) 下效果会很微弱。"
                "base_preserve 的画风幅度本身就比 interpolate 小，建议 strength >= 0.7。",
                float(strength),
            )

        # strength > 1.0 进入外推模式（CFG-style），不是错误但需要用户意识到
        if float(strength) > 1.0:
            logger.info(
                "[AnimaCrossAttn] strength=%.2f > 1.0 进入外推模式："
                "out = base + %.2f × (artist - base)。画风被放大，%s。",
                float(strength), float(strength),
                "推荐范围 1.5-2.5" if float(strength) <= 3.0 else "当前值偏高容易过饱和",
            )


        if not normalize_w and n > 1 and combine_mode in (COMBINE_OUTPUT_AVG, COMBINE_LOWRANK_AVG):
            effective_weight_sum = sum(abs(w) for w in user_weights)
            if effective_weight_sum >= 4.0 and not has_explicit_weights:
                raise ValueError(
                    f"[AnimaCrossAttn] normalize_weights=False 且画师数={n} "
                    f"(有效权重和={effective_weight_sum:.2f}，远超合理范围)。"
                    f"当前 combine={combine_mode} 下这会使 cross-attn 输出被明显放大，"
                    f"几乎必崩。\n"
                    f"改进方案 (任选一)：\n"
                    f"  1) AnimaArtistOptions 里把 normalize_weights 改为 True（推荐）\n"
                    f"  2) AnimaArtistPack 的 artist_chain 中用 ::name::0.25 调低线性强度\n"
                    f"  3) combine_mode 改用 concat (不走加权和逻辑)"
                )
            elif effective_weight_sum > 1.5:
                logger.warning(
                    "[AnimaCrossAttn] normalize_weights=False 且有效权重和=%.2f (artists=%d, combine=%s)，"
                    "cross-attn 输出可能过强。如出问题请降低 ::weight、启用 normalize 或改用 concat。",
                    effective_weight_sum, n, combine_mode,
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
            "has_explicit_weights": has_explicit_weights,
            "preset_name": preset_name,
            "artist_ema_alpha": artist_ema_alpha,
            "lowrank_k": lowrank_k,
            "artist_static_capture": artist_static_capture,
            "static_capture_k": static_capture_k,
            "artist_anchor_q": artist_anchor_q,
            "anchor_seeds_count": anchor_seeds_count,
            "anchor_user_blend": anchor_user_blend,
            "anchor_deep_layer_threshold": anchor_deep_layer_threshold,
            "individuals": None,

            "real_lens": None,
            "dm_ref": dm,
            "sigma_range": sigma_range,
            "current_sigma": None,
            "_ema_cache": {},
            "_ema_last_sigma": None,
            "_static_cache": {},
            "_static_max_sigma": None,
            "_anchor_cache": {},
            "_anchor_cache_key": None,
            "_in_anchor_run": False,
            "_anchor_failed": False,
        }



        if need_sigma_capture:
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
    "AnimaArtistPreset": AnimaArtistPreset,
    "AnimaArtistInspector": AnimaArtistInspector,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AnimaArtistPack": "Anima Artist Pack (Split + Encode)",
    "AnimaArtistCrossAttn": "Anima Artist Cross-Attn (v2)",
    "AnimaArtistOptions": "Anima Artist Options (Advanced)",
    "AnimaArtistPreset": "Anima Artist Preset (One Knob)",
    "AnimaArtistInspector": "Anima Artist Inspector",
}
