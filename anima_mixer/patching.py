"""Model validation, artist preprocessing, and patch bookkeeping helpers."""

import logging

import torch

logger = logging.getLogger(__name__)


def _in_stabilizer_window(state):
    """True while the current sigma is at or above the stabilizer threshold.

    Shared by the wrapper (EMA/static/anchor gating) and the anchor pre-run.
    A missing threshold or sigma means "always active".
    """
    threshold = state.get("stabilizer_min_sigma")
    if threshold is None:
        return True
    cur = state.get("current_sigma")
    if cur is None:
        return True
    return float(cur) >= float(threshold)


def _context_fingerprint(context):
    """Content-based fingerprint for a context tensor.

    ``id()`` is unsafe here: a freed tensor's id can be reused by a new
    allocation, silently re-hitting a stale cache. Shape + dtype + a cheap
    value checksum keys caches by content instead.
    """
    if context is None or not torch.is_tensor(context):
        return None
    try:
        sample = context.detach()
        flat = sample.reshape(-1)
        # Sample up to 1024 evenly spaced elements; cheap and stable.
        step = max(1, flat.numel() // 1024)
        digest = flat[::step].to(torch.float32).sum().item()
        return (tuple(context.shape), str(context.dtype), round(digest, 3))
    except Exception:
        return (tuple(context.shape), str(context.dtype), None)


def _forward_fingerprint(st, context):
    """Per-forward fingerprint so several forwards at the same sigma (multiple
    positive conds, regional prompts, VRAM-split batches) keep independent
    stabilizer caches instead of cross-contaminating.

    Memoized by ``id(context)`` within a run to avoid recomputing the digest
    for every layer of one forward; the memo is cleared at run start.
    """
    if context is None:
        return None
    memo = st.setdefault("_ctx_fp_memo", {})
    key = id(context)
    cached = memo.get(key)
    if cached is not None:
        return cached
    fp = _context_fingerprint(context)
    memo[key] = fp
    return fp


def reset_run_state(state):
    """Clear per-run caches and one-shot warnings at the start of a run.

    Called from the sigma-capture wrapper when sigma jumps upward (a new
    sampling pass). The content-keyed anchor caches
    (``_anchor_cache``/``_anchor_base_cache``/``_anchor_cache_key``) survive
    on purpose: the same prompt across seeds shares a fingerprint and reuses
    them. Everything that accumulates within a single pass is reset here.

    The probe accumulators are cleared in place so the probe registry's
    reference sees the reset instead of pointing at an orphaned dict.

    Known limitation: restart-style samplers jump sigma upward mid-run, so
    each restart segment resets too (stabilizers re-accumulate and the probe
    reports only the final segment). That matches the pre-existing EMA reset
    semantics and is preferable to leaking state across queue runs.
    """
    state["_disabled_layers"] = set()
    state["_disable_batched"] = False
    state["_warned_batched"] = False
    state["_warned"] = False
    state["_warned_svd"] = False
    state["_ema_cache"] = {}
    state["_static_cache"] = {}
    state["_ctx_fp_memo"] = {}
    state["_anchor_failed"] = False
    probe_stats = state.get("probe_stats")
    if isinstance(probe_stats, dict):
        probe_stats.clear()
    probe_step_stats = state.get("probe_step_stats")
    if isinstance(probe_step_stats, dict):
        probe_step_stats.clear()
    probe_seen = state.get("_probe_seen_sigmas")
    if isinstance(probe_seen, set):
        probe_seen.clear()
    if "_probe_forward_count" in state:
        state["_probe_forward_count"] = 0


def extract_conditioning(conditioning):
    """Pull (raw_embedding, t5xxl_ids, t5xxl_weights) out of a CONDITIONING."""
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


def unwrap_cross_attn(ca):
    # Imported lazily to avoid a circular import at module load time.
    from .wrapper import CrossAttnWrapper
    while isinstance(ca, CrossAttnWrapper):
        ca = ca.original
    return ca


class CrossAttnForwardPatch:
    """Callable object patch for ``cross_attn.forward``.

    Replacing the full attention module registers the wrapped module under a
    new ``.original`` state-dict path. ComfyUI can then try to restore keys like
    ``cross_attn.original.*`` after the wrapper has been removed. Patching only
    ``forward`` keeps the original module in the model tree, so parameter names
    and object-patch backups stay stable across sampler branches.
    """

    _anima_artist_mixer_forward_patch = True

    def __init__(self, wrapper):
        self.wrapper = wrapper
        self.original_forward = wrapper.original

    def __call__(self, *args, **kwargs):
        return self.wrapper.forward(*args, **kwargs)


def unwrap_cross_attn_forward(ca):
    forward = getattr(ca, "forward", None)
    while isinstance(forward, CrossAttnForwardPatch):
        forward = forward.original_forward
    return forward


def make_cross_attn_forward_patch(wrapper):
    return CrossAttnForwardPatch(wrapper)


def validate_model(diffusion_model):
    if not hasattr(diffusion_model, "blocks"):
        return False, 0, 0, f"{type(diffusion_model).__name__} has no .blocks"
    blocks = diffusion_model.blocks
    if len(blocks) == 0:
        return False, 0, 0, ".blocks is empty"
    b0 = blocks[0]
    if not hasattr(b0, "cross_attn"):
        return False, 0, 0, "blocks[0] has no cross_attn"
    ca = unwrap_cross_attn(b0.cross_attn)
    if not hasattr(ca, "context_dim"):
        return False, 0, 0, "cross_attn has no context_dim"
    return True, len(blocks), int(ca.context_dim), "ok"


def describe_external_cross_attn_patches(dm, target_blocks):
    from .wrapper import CrossAttnWrapper
    hints = []
    if not hasattr(dm, "blocks"):
        return hints
    for idx in target_blocks or []:
        if idx < 0 or idx >= len(dm.blocks):
            continue
        blk = dm.blocks[idx]
        if not hasattr(blk, "cross_attn"):
            continue
        ca = blk.cross_attn
        if isinstance(ca, CrossAttnWrapper):
            continue
        original = getattr(ca, "original", None)
        if original is None:
            continue
        hints.append(
            f"L{idx}: {type(ca).__name__} wraps {type(original).__name__}"
        )
    return hints


def preprocess_one(dm, raw, ids, weights, target_device, target_dtype):
    """Run one artist's raw embedding through Anima's LLMAdapter."""
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


def build_artists(state, ref_context):
    """Lazily preprocess every artist conditioning on first forward.

    ``real_lens`` records each artist's true token count before Anima's
    512-token zero pad. It is diagnostic only: trimming the concat K/V to it
    was tried and reverted — see the note on ``wrapper._combine_concat``.
    """
    if state.get("individuals") is not None:
        return state["individuals"], state["real_lens"]
    dm = state["dm_ref"]
    individuals, real_lens = [], []
    for raw, ids, w_t in zip(state["raws"], state["ids_list"], state["w_list"]):
        artist = preprocess_one(dm, raw, ids, w_t, ref_context.device, ref_context.dtype)
        individuals.append(artist)
        real_lens.append(int(ids.shape[-1]) if ids is not None else artist.shape[1])
    state["individuals"] = individuals
    state["real_lens"] = real_lens
    return individuals, real_lens


def broadcast_batch(t, batch_size):
    if t.shape[0] == batch_size:
        return t
    if t.shape[0] == 1:
        return t.expand(batch_size, -1, -1)
    if batch_size % t.shape[0] == 0:
        return t.repeat(batch_size // t.shape[0], 1, 1)
    return t[:1].expand(batch_size, -1, -1)


def resolve_mask(cou, batch_size, apply_to_uncond, state):
    """Build a per-row injection mask from ComfyUI's cond_or_uncond marker.

    ComfyUI may batch several latents per cond entry, in which case
    ``len(cond_or_uncond) < batch_size`` and rows are grouped in contiguous
    chunks (all rows of cond entry 0 first, then entry 1, ...). Expanding the
    markers over those chunks keeps CFG intact instead of falling back to
    injecting into every row (which would also style the uncond pass).
    """
    if apply_to_uncond:
        return [True] * batch_size
    if cou is not None and len(cou) > 0:
        if len(cou) == batch_size:
            return [c == 0 for c in cou]
        if batch_size % len(cou) == 0:
            chunk = batch_size // len(cou)
            mask = []
            for c in cou:
                mask.extend([c == 0] * chunk)
            return mask
    if not state.get("_warned", False):
        logger.warning(
            "[AnimaCrossAttn] cond_or_uncond markers unusable (got=%s, batch=%d); "
            "falling back to injecting into every row. CFG guidance may weaken — "
            "check for conflicting model patches.", cou, batch_size,
        )
        state["_warned"] = True
    return [True] * batch_size


def in_sigma_range(state):
    rng = state.get("sigma_range")
    if rng is None:
        return True
    cur = state.get("current_sigma")
    if cur is None:
        return True
    lo, hi = rng
    return lo <= cur <= hi
