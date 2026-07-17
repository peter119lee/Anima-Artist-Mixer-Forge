"""Probe nodes: per-artist/per-layer influence measurement and report."""

import logging
import uuid

import torch

from .anchor import make_sigma_capture
from .chain_tools import format_layer_span
from .constants import (
    COMBINE_OUTPUT_AVG,
    FUSION_INTERPOLATE,
    STATIC_CAPTURE_K_DEFAULT,
)
from .probe_stats import contribution_shares, render_step_curves, share_verdict
from .patching import (
    extract_conditioning,
    make_cross_attn_forward_patch,
    unwrap_cross_attn,
    unwrap_cross_attn_forward,
    validate_model,
)
from .wrapper import CrossAttnWrapper

from .nodes_core import ANY_TYPE, _build_runtime_state

logger = logging.getLogger(__name__)


# Probe results live here between graph executions, keyed by probe_id. Only
# a slim view of the runtime state is stored (the stats containers, shared
# by reference with the live wrappers) — never the full state, which would
# pin the diffusion model and artist tensors in memory across runs.
PROBE_REGISTRY = {}
_PROBE_REGISTRY_LIMIT = 8


def _registry_store(probe_id, state):
    PROBE_REGISTRY[probe_id] = {
        "probe_stats": state["probe_stats"],
        "probe_labels": state["probe_labels"],
        "probe_num_blocks": state["probe_num_blocks"],
        "_probe_seen_sigmas": state["_probe_seen_sigmas"],
        "probe_step_stats": state.get("probe_step_stats"),
        # Lets the report skip the dominance tip when the run already
        # had the contribution balancer on.
        "contribution_balance": bool(state.get("contribution_balance", False)),
    }
    while len(PROBE_REGISTRY) > _PROBE_REGISTRY_LIMIT:
        PROBE_REGISTRY.pop(next(iter(PROBE_REGISTRY)))


class AnimaArtistProbe:
    """Measure per-artist, per-layer style influence during one sampling run.

    Patches the model in probe mode: every layer records the relative delta
    norm ``||artist_out - base_out|| / ||base_out||`` for each artist while
    the image generates from the base prompt only (injection is NOT applied,
    so the measurement reflects the unmixed trajectory). Read the results
    with AnimaArtistProbeReport after the sampler has run.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "artist_pack": ("ANIMA_PACK",),
                "probe_steps": (
                    "INT",
                    {
                        "default": 6,
                        "min": 1,
                        "max": 24,
                        "step": 1,
                        "tooltip": (
                            "Number of sampling steps to measure. Early steps carry "
                            "the most style signal; 4-8 is usually enough."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING", "STRING")
    RETURN_NAMES = ("model", "base_prompt", "probe_id")
    FUNCTION = "probe"
    CATEGORY = "Anima/Diagnostics"

    def probe(self, model, artist_pack, probe_steps=6):
        if not isinstance(artist_pack, dict):
            raise ValueError(
                "[AnimaArtistProbe] artist_pack has the wrong type; connect the "
                "output of an AnimaArtistPack node"
            )
        base_cond_out = artist_pack.get("base_conditioning")
        if base_cond_out is None:
            raise ValueError("[AnimaArtistProbe] artist_pack is missing base_conditioning.")
        conditionings = artist_pack.get("conditionings") or []
        labels = list(artist_pack.get("labels") or [])
        if not conditionings:
            raise ValueError("[AnimaArtistProbe] artist_pack has no artists to probe.")

        raws, ids_list, w_list = [], [], []
        for idx, c in enumerate(conditionings):
            raw, ids, w = extract_conditioning(c)
            if raw is None:
                label = labels[idx] if idx < len(labels) else f"#{idx}"
                raise ValueError(f"[AnimaArtistProbe] artist[{label}] conditioning is empty.")
            raws.append(raw)
            ids_list.append(ids)
            w_list.append(w)

        try:
            dm = model.get_model_object("diffusion_model")
        except Exception:
            dm = model.model.diffusion_model

        ok, num_blocks, _, msg = validate_model(dm)
        if not ok:
            raise ValueError(f"[AnimaArtistProbe] unsupported model: {msg}")
        if not hasattr(dm, "preprocess_text_embeds"):
            raise ValueError("[AnimaArtistProbe] this is not an Anima model (missing preprocess_text_embeds)")

        probe_id = uuid.uuid4().hex[:12]
        n = len(raws)

        state = _build_runtime_state(
            True,
            FUSION_INTERPOLATE,
            COMBINE_OUTPUT_AVG,
            1.0,
            False,
            raws,
            ids_list,
            w_list,
            [1.0] * n,
            labels,
            [None] * n,
            False,
            [None] * n,
            False,
            True,
            False,
            None,
            {"static_capture_k": STATIC_CAPTURE_K_DEFAULT},
            dm,
            None,
            [],
        )
        state["probe_steps"] = max(1, int(probe_steps))
        state["probe_stats"] = {}  # layer_idx -> [ [sum, count], ... ] per artist
        state["probe_step_stats"] = {}  # sigma_key -> [ [sum, count], ... ] per artist
        state["probe_labels"] = labels
        state["probe_num_blocks"] = num_blocks
        state["_probe_seen_sigmas"] = set()
        # Fallback step budget for the degenerate case where the sigma-capture
        # hook was overridden and current_sigma is never set.
        state["_probe_forward_count"] = 0

        m = model.clone()
        prev = m.model_options.get("model_function_wrapper")
        m.set_model_unet_function_wrapper(make_sigma_capture(state, prev))
        for i in range(num_blocks):
            inner = unwrap_cross_attn_forward(unwrap_cross_attn(dm.blocks[i].cross_attn))
            wrapper = _ProbeCrossAttnWrapper(inner, state, i)
            m.add_object_patch(
                f"diffusion_model.blocks.{i}.cross_attn.forward",
                make_cross_attn_forward_patch(wrapper),
            )

        _registry_store(probe_id, state)
        logger.info(
            "[AnimaArtistProbe] probe %s armed for %d artists x %d layers (first %d steps)",
            probe_id,
            n,
            num_blocks,
            state["probe_steps"],
        )
        return (m, base_cond_out, probe_id)


class _ProbeCrossAttnWrapper(CrossAttnWrapper):
    """Measurement-only wrapper: never alters the output."""

    def _dispatch(self, x, context, rope_emb, transformer_options):
        st = self._st
        base_out = self.original(x, context, rope_emb=rope_emb, transformer_options=transformer_options)

        stats = st.setdefault("probe_stats", {})
        layer_stats = stats.get(self._idx)
        n = len(st["raws"])
        if layer_stats is None:
            layer_stats = [[0.0, 0] for _ in range(n)]
            stats[self._idx] = layer_stats
        # Measure only the first probe_steps distinct sigmas; afterwards the
        # forward is a plain pass-through (and the seen-set stops growing so
        # the report's step count stays accurate). When the sigma is missing
        # (capture hook overridden), fall back to a raw forward counter so the
        # budget is still enforced instead of measuring forever.
        budget = int(st.get("probe_steps", 6))
        cur = st.get("current_sigma")
        if cur is not None:
            seen = st.setdefault("_probe_seen_sigmas", set())
            cur_key = round(float(cur), 4)
            if cur_key not in seen:
                if len(seen) >= budget:
                    return base_out
                seen.add(cur_key)
        else:
            count = int(st.get("_probe_forward_count", 0))
            if count >= budget:
                return base_out
            st["_probe_forward_count"] = count + 1

        # Restrict the delta measurement to the cond rows: under CFG the uncond
        # rows carry the unstyled trajectory and would understate influence.
        from .patching import build_artists, resolve_mask

        cou = transformer_options.get("cond_or_uncond") if isinstance(transformer_options, dict) else None
        mask = resolve_mask(cou, context.shape[0], False, {})
        row_mask = torch.tensor(mask, device=base_out.device, dtype=torch.bool)

        individuals, _ = build_artists(st, context)
        outs = self._collect_artist_outputs(
            x,
            context,
            rope_emb,
            transformer_options,
            individuals,
            FUSION_INTERPOLATE,
        )
        base_sel = base_out.detach().to(torch.float32)[row_mask]
        base_norm = float(base_sel.norm().item())
        if base_norm <= 1e-8:
            return base_out
        # Per-step accumulation (v27.2 curves) mirrors the layer accumulation;
        # skipped when the sigma is unknown (capture hook overridden).
        step_row = None
        if cur is not None:
            step_stats = st.get("probe_step_stats")
            if isinstance(step_stats, dict):
                step_key = round(float(cur), 4)
                step_row = step_stats.get(step_key)
                if step_row is None:
                    step_row = [[0.0, 0] for _ in range(n)]
                    step_stats[step_key] = step_row
        for i, out_i in enumerate(outs):
            delta = out_i.detach().to(torch.float32)[row_mask] - base_sel
            rel = float(delta.norm().item()) / base_norm
            layer_stats[i][0] += rel
            layer_stats[i][1] += 1
            if step_row is not None:
                step_row[i][0] += rel
                step_row[i][1] += 1
        return base_out


def _suggest_layer_range(scores, top_fraction=0.35):
    """Suggest a contiguous layer range covering the strongest layers."""
    if not scores:
        return None
    indexed = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    keep = max(1, int(round(len(scores) * top_fraction)))
    top = sorted(indexed[:keep])
    return top[0], top[-1]


class AnimaArtistProbeReport:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "probe_id": ("STRING", {"default": "", "forceInput": True}),
            },
            "optional": {
                "trigger": (
                    ANY_TYPE,
                    {
                        "tooltip": (
                            "Connect any post-sampler output (e.g. the decoded IMAGE) "
                            "so this report runs after sampling finished."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("report",)
    FUNCTION = "report"
    CATEGORY = "Anima/Diagnostics"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, probe_id, trigger=None):
        return float("nan")  # always re-run; registry contents change

    def report(self, probe_id, trigger=None):
        state = PROBE_REGISTRY.get(str(probe_id or "").strip())
        if state is None:
            text = (
                "Anima Artist Probe Report\n\n"
                f"status: NO DATA\nprobe_id: {probe_id!r}\n\n"
                "No measurements found. Wire AnimaArtistProbe's model output "
                "through the sampler, connect its probe_id here, and connect "
                "a post-sampler output (e.g. IMAGE) to trigger."
            )
            return {"ui": {"text": [text]}, "result": (text,)}

        stats = state.get("probe_stats") or {}
        labels = list(state.get("probe_labels") or [])
        num_blocks = int(state.get("probe_num_blocks", 0))
        if not stats:
            text = (
                "Anima Artist Probe Report\n\n"
                "status: EMPTY\n\n"
                "The probe is armed but no samples were recorded yet. "
                "Run the sampler first (connect trigger to a post-sampler output)."
            )
            return {"ui": {"text": [text]}, "result": (text,)}

        n = len(labels)
        # scores[artist][layer] = mean relative delta
        scores = [[0.0] * num_blocks for _ in range(n)]
        sample_counts = [0] * n
        for layer_idx, layer_stats in stats.items():
            for i, (total, count) in enumerate(layer_stats):
                if i < n and 0 <= layer_idx < num_blocks and count > 0:
                    scores[i][layer_idx] = total / count
                    sample_counts[i] = max(sample_counts[i], count)

        lines = [
            "Anima Artist Probe Report",
            "",
            "status: OK",
            f"artists: {n}",
            f"layers: {num_blocks}",
            f"measured steps: {len(state.get('_probe_seen_sigmas') or [])}",
        ]
        totals, shares = contribution_shares(scores)
        lines.append("")
        lines.append("contribution split (share of summed mean influence):")
        for i, label in enumerate(labels):
            lines.append(
                f"  {label}: {shares[i] * 100:.1f}% "
                f"({shares[i] * n:.2f}x equal split) — {share_verdict(shares[i], n)}"
            )
        has_dominant = any(share_verdict(s, n) == "dominant" for s in shares)
        if has_dominant and not state.get("contribution_balance", False):
            lines.append(
                "  tip: enable contribution_balance (Options node) or add the "
                "Style Balance node to even artist strength before weighting"
            )
        curve_lines = render_step_curves(state.get("probe_step_stats"), labels)
        if curve_lines:
            lines.append("")
            lines.extend(curve_lines)
        lines.extend(
            [
                "",
                "relative style influence per layer (||artist_out - base_out|| / ||base_out||):",
            ]
        )
        for i, label in enumerate(labels):
            row = scores[i]
            peak = max(row) if row else 0.0
            lines.append("")
            lines.append(f"artist {i + 1}: {label} (peak {peak:.3f}, {sample_counts[i]} samples)")
            # Compact bar chart, 8 layers per line.
            for start in range(0, num_blocks, 8):
                seg = row[start : start + 8]
                cells = []
                for j, v in enumerate(seg):
                    bar_len = 0 if peak <= 0 else int(round(6 * v / peak))
                    cells.append(f"L{start + j:>2}:{'#' * bar_len:<6}")
                lines.append("  " + " ".join(cells))
            suggestion = _suggest_layer_range(row)
            if suggestion is not None:
                lo, hi = suggestion
                lines.append(
                    f"  suggested route: {label}@{lo}-{hi}  "
                    f"({format_layer_span(lo, hi)} carries the strongest signal)"
                )
        lines.extend(
            [
                "",
                "how to use:",
                "  - copy the suggested @routes into your artist chain",
                "  - artists with flat profiles mix well at all layers",
                "  - artists with sharp peaks benefit most from layer routing",
            ]
        )
        text = "\n".join(lines)
        return {"ui": {"text": [text]}, "result": (text,)}
