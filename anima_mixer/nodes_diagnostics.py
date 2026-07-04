"""Diagnostics nodes (v27.1): is each artist actually working, and how much?

Three complementary levels of evidence:

- ``AnimaArtistTagCheck``   encoder level, free: compares each artist's
  conditioning against the base conditioning and against the other artists
  straight out of the pack. Reliably catches duplicate/alias entries and
  no-op entries; it can NOT tell known tags from unknown ones (validated on
  live Anima: gibberish and real artists overlap in encoder shift).
- ``AnimaArtistABVariants`` run level: emits a list of artist-chain variants
  (off / full / solo / leave-one-out / cumulative) so one queue renders a
  same-seed comparison series via ComfyUI's list fan-out.
- ``AnimaArtistImpactMap``  image level: diff heatmap plus impact metrics
  between two same-seed renders (e.g. mixer off vs on).

No ComfyUI imports here: the pytest suite runs without a live ComfyUI.
"""

import logging

import torch
import torch.nn.functional as F

from .constants import MAX_ARTISTS
from .parsing import (
    parse_artist_entries,
    parse_artist_layer_routes,
    parse_artist_timing_routes,
    split_artist_chain,
)
from .patching import extract_conditioning

logger = logging.getLogger(__name__)

# Thresholds calibrated against live Anima encodes (2026-07-04): real artist
# tags shift the pooled encoding by ~0.013-0.039 and OVERLAP with gibberish
# tags (0.015-0.035), so no shift threshold can detect unknown tags. Only the
# mathematical extremes carry a verdict; the report prints raw numbers.
TAG_NOOP_DIST = 1e-4      # shift vs base below this => entry adds ~nothing
TAG_DUP_SIM = 0.999       # pairwise cosine above this => near-duplicate pair

IMPACT_CHANGE_THRESHOLD = 0.04   # per-pixel mean-abs diff counted as "changed"
IMPACT_AUTO_GAIN_MAX = 10000.0

_HEAT_STOPS = (
    (0.00, (0.00, 0.00, 0.00)),
    (0.25, (0.15, 0.00, 0.35)),
    (0.50, (0.85, 0.20, 0.05)),
    (0.75, (1.00, 0.65, 0.00)),
    (1.00, (1.00, 1.00, 0.85)),
)


def _pool_conditioning(raw):
    """Mean-pool a [B, T, D] (or [T, D]) embedding to a single [D] vector."""
    t = raw.detach().float()
    if t.dim() < 2:
        return None
    return t.reshape(-1, t.shape[-1]).mean(dim=0).cpu()


def _cosine(a, b):
    denom = float(a.norm()) * float(b.norm())
    if denom < 1e-12:
        return 0.0
    return float(torch.dot(a, b)) / denom


class AnimaArtistTagCheck:
    """Encoder-level duplicate / no-op detector: no sampling, no extra encodes."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "artist_pack": ("ANIMA_PACK", {
                    "tooltip": (
                        "Output of AnimaArtistPack. The check reuses the "
                        "conditionings already encoded there; it costs no "
                        "sampling and no extra CLIP passes."
                    ),
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("report",)
    FUNCTION = "check"
    CATEGORY = "Anima/CrossAttn"
    OUTPUT_NODE = True

    def check(self, artist_pack):
        if not isinstance(artist_pack, dict):
            raise ValueError(
                "[AnimaArtistTagCheck] artist_pack has the wrong type; connect "
                "the output of an AnimaArtistPack node"
            )
        base_raw, _, _ = extract_conditioning(artist_pack.get("base_conditioning"))
        if base_raw is None:
            raise ValueError(
                "[AnimaArtistTagCheck] artist_pack is missing base_conditioning."
            )
        base_vec = _pool_conditioning(base_raw)
        if base_vec is None:
            raise ValueError("[AnimaArtistTagCheck] base conditioning has no token axis.")

        conditionings = artist_pack.get("conditionings") or []
        labels = list(artist_pack.get("labels") or [])
        weights = list(artist_pack.get("weights") or [])
        if not conditionings:
            report = (
                "[AnimaArtistTagCheck] no artists in the pack (empty chain); "
                "nothing to check."
            )
            return {"ui": {"text": [report]}, "result": (report,)}

        base_norm = float(base_vec.norm())
        entries = []
        for idx, cond in enumerate(conditionings):
            label = labels[idx] if idx < len(labels) else f"#{idx}"
            raw, _, _ = extract_conditioning(cond)
            vec = _pool_conditioning(raw) if raw is not None else None
            if vec is None:
                raise ValueError(
                    f"[AnimaArtistTagCheck] artist[{label}] conditioning is empty."
                )
            sim = _cosine(vec, base_vec)
            delta = float((vec - base_vec).norm()) / max(base_norm, 1e-12)
            weight = float(weights[idx]) if idx < len(weights) else 1.0
            entries.append({
                "label": label, "vec": vec, "sim": sim,
                "dist": max(0.0, 1.0 - sim), "delta": delta, "weight": weight,
            })

        duplicates = []
        for i in range(len(entries)):
            for j in range(i + 1, len(entries)):
                if _cosine(entries[i]["vec"], entries[j]["vec"]) >= TAG_DUP_SIM:
                    duplicates.append((i, j))

        dup_members = {idx for pair in duplicates for idx in pair}
        lines = [
            "[AnimaArtistTagCheck] encoder-level distinctiveness "
            f"({len(entries)} artists; shift = 1 - cosine vs base prompt):",
        ]
        for idx, e in enumerate(entries):
            flags = []
            if e["dist"] < TAG_NOOP_DIST:
                flags.append("[NO-OP]")
            if idx in dup_members:
                flags.append("[DUPLICATE]")
            if not flags:
                flags.append("[OK]")
            lines.append(
                f"  {' '.join(flags)} {e['label']} — shift {e['dist']:.4f}, "
                f"delta-norm {e['delta']:.3f}x base, weight {e['weight']:.2f}"
            )
        for i, j in duplicates:
            lines.append(
                f"  [DUPLICATE] '{entries[i]['label']}' and '{entries[j]['label']}' "
                "encode almost identically; mixing them adds no second style."
            )
        lines.append(
            "  legend: [NO-OP] encodes identically to the base prompt; "
            "[DUPLICATE] two entries carry the same style vector (repeat or "
            "alias). Encoder shift can NOT tell known tags from unknown ones "
            "(gibberish and real artists overlap on live Anima) — use "
            "AnimaArtistABVariants or the Layer Probe to see whether an "
            "artist actually changes the image."
        )
        report = "\n".join(lines)
        for i, j in duplicates:
            logger.warning(
                "[AnimaArtistTagCheck] '%s' and '%s' encode ~identically "
                "(duplicate or alias)", entries[i]["label"], entries[j]["label"],
            )
        return {"ui": {"text": [report]}, "result": (report,)}


def sanitize_label(text):
    """Filename-safe variant label (Windows-forbidden chars stripped)."""
    out = []
    for ch in str(text or ""):
        out.append(ch if (ch.isalnum() or ch in "._-") else "_")
    label = "".join(out)
    while "__" in label:
        label = label.replace("__", "_")
    label = label.strip("._ ") or "artist"
    return label[:48]


def _chain_names(parts):
    """Clean artist names for labeling, via the same pipeline the Pack uses."""
    stripped, _ = parse_artist_timing_routes(list(parts))
    stripped, _ = parse_artist_layer_routes(stripped)
    entries = parse_artist_entries(stripped)
    names = []
    for idx, entry in enumerate(entries):
        name = str(entry[0] or "").strip()
        names.append(name or f"artist{idx + 1}")
    return names


def build_variants(artist_chain, mode, include_no_mixer, include_full_mix):
    """Return (chains, labels, report) for one A/B comparison series."""
    parts = split_artist_chain(artist_chain)
    names = _chain_names(parts)
    full_chain = ", ".join(parts)
    warnings = []

    raw = []  # (chain_text, label_stem)
    if not parts:
        warnings.append("artist_chain is empty; emitting only the no-mixer baseline.")
        raw.append(("", "no_mixer"))
    elif mode == "off_vs_full":
        raw.append(("", "no_mixer"))
        raw.append((full_chain, "full_mix"))
    else:
        if include_no_mixer:
            raw.append(("", "no_mixer"))
        if mode == "cumulative":
            # The ramp ends at the full mix; a separate full baseline would
            # only duplicate the last step.
            for i, name in enumerate(names):
                raw.append((", ".join(parts[: i + 1]), f"add_{name}"))
        else:
            if include_full_mix:
                raw.append((full_chain, "full_mix"))
            if mode == "solo_each":
                for part, name in zip(parts, names):
                    raw.append((part, f"solo_{name}"))
            elif mode == "leave_one_out":
                for i, name in enumerate(names):
                    rest = parts[:i] + parts[i + 1:]
                    raw.append((", ".join(rest), f"without_{name}"))
            else:
                raise ValueError(f"[AnimaArtistABVariants] unknown mode: {mode!r}")

    chains, labels, seen = [], [], {}
    for chain_text, stem in raw:
        if chain_text in seen:
            warnings.append(
                f"skipped duplicate variant '{stem}' (same chain as '{seen[chain_text]}')."
            )
            continue
        seen[chain_text] = stem
        chains.append(chain_text)
        labels.append(f"{len(chains):02d}_{sanitize_label(stem)}")

    if len(parts) > MAX_ARTISTS:
        warnings.append(
            f"chain has {len(parts)} artists; AnimaArtistPack truncates past {MAX_ARTISTS}."
        )

    lines = [
        f"[AnimaArtistABVariants] {len(chains)} variants ({mode}); each one is a "
        "full same-seed sampling run:",
    ]
    for chain_text, label in zip(chains, labels):
        lines.append(f"  {label}: {chain_text or '(no artists)'}")
    lines.extend(f"  note: {w}" for w in warnings)
    lines.append(
        "  wiring: artist_chain -> AnimaArtistPack.artist_chain, "
        "label -> SaveImage.filename_prefix; keep the sampler seed fixed."
    )
    return chains, labels, "\n".join(lines)


class AnimaArtistABVariants:
    """Fan out one artist chain into a same-seed A/B comparison series."""

    MODES = ("solo_each", "leave_one_out", "cumulative", "off_vs_full")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "artist_chain": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": (
                        "Same syntax as AnimaArtistPack. Weights, @layers and "
                        "%timing routes stay attached to their artist in every "
                        "variant."
                    ),
                }),
                "mode": (list(cls.MODES), {
                    "default": "solo_each",
                    "tooltip": (
                        "solo_each: one variant per artist alone.\n"
                        "leave_one_out: full mix minus one artist each.\n"
                        "cumulative: add artists one by one.\n"
                        "off_vs_full: just baseline vs full mix."
                    ),
                }),
                "include_no_mixer": ("BOOLEAN", {"default": True}),
                "include_full_mix": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("artist_chain", "label", "report")
    OUTPUT_IS_LIST = (True, True, False)
    FUNCTION = "build"
    CATEGORY = "Anima/CrossAttn"
    OUTPUT_NODE = True

    def build(self, artist_chain, mode, include_no_mixer, include_full_mix):
        chains, labels, report = build_variants(
            artist_chain, mode, bool(include_no_mixer), bool(include_full_mix)
        )
        logger.info("[AnimaArtistABVariants] emitting %d variants (%s)", len(chains), mode)
        return {"ui": {"text": [report]}, "result": (chains, labels, report)}


def _as_bhwc(t, name):
    if not torch.is_tensor(t):
        raise ValueError(f"[AnimaArtistImpactMap] {name} is not an image tensor.")
    x = t.detach().float().cpu()
    if x.dim() == 3:
        x = x.unsqueeze(0)
    if x.dim() != 4:
        raise ValueError(f"[AnimaArtistImpactMap] {name} must be [B,H,W,C].")
    return x


def _to_rgb(x):
    if x.shape[-1] == 3:
        return x
    if x.shape[-1] == 1:
        return x.repeat(1, 1, 1, 3)
    return x[..., :3]


def _luminance(x):
    if x.shape[-1] >= 3:
        r, g, b = x[..., 0], x[..., 1], x[..., 2]
        return 0.2126 * r + 0.7152 * g + 0.0722 * b
    return x.mean(dim=-1)


def _blur(x_bhwc, kernel):
    n = x_bhwc.permute(0, 3, 1, 2)
    pad = kernel // 2
    return F.avg_pool2d(n, kernel, stride=1, padding=pad,
                        count_include_pad=False).permute(0, 2, 3, 1)


def _heat_colors(norm):
    """Map a [B,H,W] magnitude in [0,1] to an RGB heatmap [B,H,W,3]."""
    out = torch.zeros(norm.shape + (3,), dtype=torch.float32)
    for (lo, lo_rgb), (hi, hi_rgb) in zip(_HEAT_STOPS[:-1], _HEAT_STOPS[1:]):
        mask = (norm >= lo) & (norm <= hi) if hi >= 1.0 else (norm >= lo) & (norm < hi)
        if not bool(mask.any()):
            continue
        t = ((norm - lo) / max(hi - lo, 1e-12)).clamp(0.0, 1.0)[mask]
        for c in range(3):
            out[..., c][mask] = lo_rgb[c] + (hi_rgb[c] - lo_rgb[c]) * t
    return out


class AnimaArtistImpactMap:
    """Where and how strongly two same-seed renders differ."""

    LAYOUTS = ("triptych", "overlay", "heatmap")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_a": ("IMAGE", {"tooltip": "Reference render (e.g. mixer off)."}),
                "image_b": ("IMAGE", {"tooltip": "Test render (e.g. mixer on), same seed/size."}),
                "layout": (list(cls.LAYOUTS), {
                    "default": "triptych",
                    "tooltip": (
                        "triptych: [A | B | change overlay]. overlay: grayscale B "
                        "with the change heatmap on top. heatmap: heatmap only."
                    ),
                }),
            },
            "optional": {
                "auto_gain": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "Scale the heatmap so the strongest change is fully "
                        "visible. Disable (fixed gain) to compare heatmaps "
                        "across different runs on one scale."
                    ),
                }),
                "gain": ("FLOAT", {
                    "default": 4.0, "min": 0.5, "max": 100.0, "step": 0.5,
                    "tooltip": "Fixed heatmap gain used when auto_gain is off.",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "FLOAT")
    RETURN_NAMES = ("visualization", "report", "impact_score")
    FUNCTION = "compare"
    CATEGORY = "Anima/CrossAttn"
    OUTPUT_NODE = True

    def compare(self, image_a, image_b, layout="triptych", auto_gain=True, gain=4.0):
        a = _as_bhwc(image_a, "image_a")
        b = _as_bhwc(image_b, "image_b")
        if a.shape[1:] != b.shape[1:]:
            raise ValueError(
                f"[AnimaArtistImpactMap] size mismatch: image_a {tuple(a.shape)} vs "
                f"image_b {tuple(b.shape)}; compare same-seed, same-size renders."
            )
        if a.shape[0] != b.shape[0]:
            if a.shape[0] == 1:
                a = a.expand(b.shape[0], -1, -1, -1)
            elif b.shape[0] == 1:
                b = b.expand(a.shape[0], -1, -1, -1)
            else:
                raise ValueError(
                    f"[AnimaArtistImpactMap] batch mismatch: {a.shape[0]} vs {b.shape[0]}."
                )

        batch, height, width, channels = a.shape
        mag = (a - b).abs().mean(dim=-1)  # [B,H,W]

        if auto_gain:
            p99 = float(torch.quantile(mag.flatten(), 0.99))
            used_gain = min(max(1.0 / max(p99, 1e-6), 1.0), IMPACT_AUTO_GAIN_MAX)
        else:
            used_gain = float(gain)
        norm = (mag * used_gain).clamp(0.0, 1.0)
        heat = _heat_colors(norm)

        kernel = max(3, min(height, width) // 16) | 1
        blur_a, blur_b = _blur(a, kernel), _blur(b, kernel)

        lines = [
            "[AnimaArtistImpactMap] A/B difference "
            f"({batch} item(s), {height}x{width}, gain {used_gain:.1f}"
            f"{' auto' if auto_gain else ' fixed'}):",
        ]
        scores = []
        for i in range(batch):
            mean_abs = float(mag[i].mean())
            scores.append(mean_abs * 100.0)
            changed = float((mag[i] > IMPACT_CHANGE_THRESHOLD).float().mean()) * 100.0
            low = float((blur_a[i] - blur_b[i]).abs().mean()) * 100.0
            high = float(((a[i] - blur_a[i]) - (b[i] - blur_b[i])).abs().mean()) * 100.0
            item = (
                f"  item {i}: impact {mean_abs * 100.0:.2f}% | changed area "
                f"{changed:.1f}% (>{IMPACT_CHANGE_THRESHOLD}) | composition(low-freq) "
                f"{low:.2f}% vs texture(high-freq) {high:.2f}%"
            )
            if channels >= 3:
                lum = float((_luminance(a[i:i + 1]) - _luminance(b[i:i + 1])).abs().mean()) * 100.0
                item += f" | luminance {lum:.2f}%"
            lines.append(item)
        score = sum(scores) / len(scores)
        if score < 0.1:
            verdict = "no visible change — the artist/setting had ~zero effect"
        elif score < 1.0:
            verdict = "subtle change (texture-level)"
        elif score < 4.0:
            verdict = "clear style shift"
        else:
            verdict = "major change (composition likely affected too)"
        lines.append(f"  verdict: {verdict}.")
        report = "\n".join(lines)

        if layout == "heatmap":
            viz = heat
        else:
            gray = (_luminance(b) * 0.45).unsqueeze(-1).expand(-1, -1, -1, 3)
            weight = (norm * 0.9).unsqueeze(-1)
            overlay = gray * (1.0 - weight) + heat * weight
            if layout == "overlay":
                viz = overlay
            else:
                viz = torch.cat(
                    [_to_rgb(a).clamp(0, 1), _to_rgb(b).clamp(0, 1), overlay], dim=2
                )
        return {
            "ui": {"text": [report]},
            "result": (viz.clamp(0.0, 1.0), report, float(score)),
        }
