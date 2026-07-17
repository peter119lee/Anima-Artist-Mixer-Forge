"""Tests for the v27.5 upstream ports (An1X3R/Anima-Artist-Mixer, MIT).

Covers the AnimaArtistStyleBalance compat shim (maps upstream's dial onto
the forge's contribution_balance controller), VRAM-aware automatic artist
chunking, Q-projection reuse with runtime validation, and the probe
report's dominance tip.

unittest.TestCase style so both pytest and unittest discovery collect them.
"""

import os
import sys
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch  # noqa: E402

from anima_mixer.nodes_core import (  # noqa: E402
    AnimaArtistProbeReport,
    _registry_store,
)
from anima_mixer.nodes_options import AnimaArtistStyleBalance  # noqa: E402
from anima_mixer.wrapper import CrossAttnWrapper  # noqa: E402


class StyleBalanceShimTests(unittest.TestCase):
    def test_dial_maps_to_contribution_balance(self):
        (opts,) = AnimaArtistStyleBalance().build(style_balance=0.6)
        self.assertTrue(opts["contribution_balance"])
        self.assertAlmostEqual(opts["contribution_balance_alpha"], 0.6)

    def test_zero_dial_is_a_passthrough(self):
        incoming = {"contribution_balance": True, "contribution_balance_alpha": 0.3}
        (opts,) = AnimaArtistStyleBalance().build(
            style_balance=0.0,
            advanced_options=incoming,
        )
        self.assertEqual(opts, incoming)
        self.assertIsNot(opts, incoming)  # never mutate the incoming dict

    def test_dial_clamped_to_unit_range(self):
        (opts,) = AnimaArtistStyleBalance().build(style_balance=5.0)
        self.assertEqual(opts["contribution_balance_alpha"], 1.0)

    def test_chains_on_existing_options(self):
        incoming = {"max_batch_artists": 2}
        (opts,) = AnimaArtistStyleBalance().build(
            style_balance=0.5,
            advanced_options=incoming,
        )
        self.assertEqual(opts["max_batch_artists"], 2)
        self.assertTrue(opts["contribution_balance"])

    def test_registered_with_upstream_node_id(self):
        # Upstream 26.x workflows reference this exact node id; the shim
        # keeps them loadable in the forge.
        from anima_mixer import NODE_CLASS_MAPPINGS

        self.assertIn("AnimaArtistStyleBalance", NODE_CLASS_MAPPINGS)


class ProbeDominanceTipTests(unittest.TestCase):
    def _state(self, dominant=True, balance_on=False):
        if dominant:
            probe_stats = {0: [[2.0, 2], [0.5, 2]], 1: [[1.0, 2], [0.5, 2]]}
        else:
            probe_stats = {0: [[1.0, 2], [1.0, 2]], 1: [[0.5, 2], [0.5, 2]]}
        state = {
            "probe_stats": probe_stats,
            "probe_labels": ["strong_artist", "weak_artist"],
            "probe_num_blocks": 2,
            "_probe_seen_sigmas": {14.0},
        }
        if balance_on:
            state["contribution_balance"] = True
        return state

    def test_dominant_verdict_appends_balance_tip(self):
        _registry_store("tip-dominant", self._state())
        report = AnimaArtistProbeReport().report("tip-dominant")["result"][0]
        self.assertIn("dominant", report)
        self.assertIn("contribution_balance", report)

    def test_no_tip_when_balance_already_enabled(self):
        _registry_store("tip-already-on", self._state(balance_on=True))
        report = AnimaArtistProbeReport().report("tip-already-on")["result"][0]
        self.assertIn("dominant", report)
        self.assertNotIn("tip:", report)

    def test_no_tip_when_split_is_balanced(self):
        _registry_store("tip-balanced", self._state(dominant=False))
        report = AnimaArtistProbeReport().report("tip-balanced")["result"][0]
        self.assertNotIn("tip:", report)


class _StubAnimaAttention(torch.nn.Module):
    """Mirrors comfy/ldm/cosmos/predict2.py Attention's cross-attn interface
    (rope only applies to self-attention there, so the stub ignores it)."""

    is_selfattn = False

    def __init__(self, dim=8, heads=2):
        super().__init__()
        self.n_heads = heads
        self.head_dim = dim // heads
        self.q_proj = torch.nn.Linear(dim, dim, bias=False)
        self.k_proj = torch.nn.Linear(dim, dim, bias=False)
        self.v_proj = torch.nn.Linear(dim, dim, bias=False)
        self.q_norm = torch.nn.Identity()
        self.k_norm = torch.nn.Identity()
        self.v_norm = torch.nn.Identity()

    def compute_attention(self, q, k, v, transformer_options=None):
        qh = q.permute(0, 2, 1, 3)
        kh = k.permute(0, 2, 1, 3)
        vh = v.permute(0, 2, 1, 3)
        attn = torch.softmax(qh @ kh.transpose(-1, -2) / self.head_dim**0.5, dim=-1)
        out = (attn @ vh).permute(0, 2, 1, 3)
        return out.reshape(*out.shape[:2], -1)

    def forward(self, x, context=None, rope_emb=None, transformer_options=None):
        q_shape = (*x.shape[:-1], self.n_heads, self.head_dim)
        kv_shape = (*context.shape[:-1], self.n_heads, self.head_dim)
        q = self.q_norm(self.q_proj(x).view(q_shape))
        k = self.k_norm(self.k_proj(context).view(kv_shape))
        v = self.v_norm(self.v_proj(context).view(kv_shape))
        return self.compute_attention(q, k, v, transformer_options=transformer_options)


class _CountingAttn(torch.nn.Module):
    """K/V-mean stub that counts forward invocations."""

    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, x, context=None, rope_emb=None, transformer_options=None):
        self.calls += 1
        mean = context.mean(dim=1, keepdim=True)
        return mean.expand(x.shape[0], x.shape[1], context.shape[-1])


class QReuseBatchingTests(unittest.TestCase):
    def test_original_module_defaults_to_none(self):
        w = CrossAttnWrapper(torch.nn.Identity(), {}, 0)
        self.assertIsNone(w.original_module)

    def test_q_reuse_off_by_default(self):
        # Live A/B showed the fp16 kernel difference shifts same-seed
        # renders, so Q reuse must never engage without artist_q_reuse.
        torch.manual_seed(3)
        stub = _StubAnimaAttention()
        state = {}
        w = CrossAttnWrapper(stub.forward, state, 0, original_module=stub)
        x = torch.randn(1, 3, 8)
        ctx = torch.randn(1, 4, 8)
        artists = [torch.randn(1, 4, 8) for _ in range(2)]

        outs = w._collect_artist_outputs(x, ctx, None, {}, artists, "interpolate")

        expected = [stub(x, a) for a in artists]
        for got, want in zip(outs, expected):
            self.assertTrue(torch.allclose(got, want, atol=1e-5))
        self.assertNotIn("_q_reuse_validation", state)

    def test_batched_matches_sequential_with_q_reuse(self):
        torch.manual_seed(0)
        stub = _StubAnimaAttention()
        state = {"artist_q_reuse": True}
        w = CrossAttnWrapper(stub.forward, state, 0, original_module=stub)
        x = torch.randn(2, 3, 8)
        ctx = torch.randn(2, 4, 8)
        artists = [torch.randn(1, 4, 8) for _ in range(3)]

        outs = w._collect_artist_outputs(x, ctx, None, {}, artists, "interpolate")

        expected = [stub(x, a.expand(2, -1, -1)) for a in artists]
        for got, want in zip(outs, expected):
            self.assertTrue(torch.allclose(got, want, atol=1e-5))
        self.assertTrue(state["_q_reuse_validation"][type(stub)])

    def test_q_reuse_stays_active_on_later_calls(self):
        torch.manual_seed(1)
        stub = _StubAnimaAttention()
        state = {"artist_q_reuse": True}
        w = CrossAttnWrapper(stub.forward, state, 0, original_module=stub)
        x = torch.randn(1, 3, 8)
        ctx = torch.randn(1, 4, 8)
        artists = [torch.randn(1, 4, 8) for _ in range(2)]

        first = w._collect_artist_outputs(x, ctx, None, {}, artists, "interpolate")
        second = w._collect_artist_outputs(x, ctx, None, {}, artists, "interpolate")

        for a, b in zip(first, second):
            self.assertTrue(torch.allclose(a, b, atol=1e-5))

    def test_module_without_internals_uses_standard_path(self):
        counting = _CountingAttn()
        state = {"artist_q_reuse": True}
        w = CrossAttnWrapper(counting.forward, state, 0, original_module=counting)
        x = torch.zeros(1, 2, 4)
        ctx = torch.full((1, 3, 4), 3.0)
        artists = [torch.ones(1, 3, 4), torch.full((1, 3, 4), 2.0)]

        outs = w._collect_artist_outputs(x, ctx, None, {}, artists, "interpolate")

        self.assertTrue(torch.allclose(outs[0], torch.ones_like(outs[0])))
        self.assertTrue(torch.allclose(outs[1], torch.full_like(outs[1], 2.0)))
        self.assertNotIn(type(counting), state.get("_q_reuse_validation", {}))

    def test_concat_fusion_batches_with_q_reuse(self):
        torch.manual_seed(2)
        stub = _StubAnimaAttention()
        state = {"artist_q_reuse": True}
        w = CrossAttnWrapper(stub.forward, state, 0, original_module=stub)
        x = torch.randn(1, 3, 8)
        ctx = torch.randn(1, 4, 8)
        artists = [torch.randn(1, 4, 8) for _ in range(2)]

        outs = w._collect_artist_outputs(
            x,
            ctx,
            None,
            {},
            artists,
            "concat_with_base",
        )

        expected = [stub(x, torch.cat([ctx, a], dim=1)) for a in artists]
        for got, want in zip(outs, expected):
            self.assertTrue(torch.allclose(got, want, atol=1e-5))


class AutoChunkTests(unittest.TestCase):
    def test_cpu_returns_full_artist_count(self):
        w = CrossAttnWrapper(torch.nn.Identity(), {}, 0)
        self.assertEqual(
            w._auto_artist_chunk_size(torch.zeros(1, 2, 4), 16, 5),
            5,
        )

    def test_manual_limit_still_chunks_batched_forwards(self):
        counting = _CountingAttn()
        w = CrossAttnWrapper(counting.forward, {"max_batch_artists": 2}, 0)
        x = torch.zeros(1, 2, 4)
        ctx = torch.full((1, 3, 4), 3.0)
        artists = [torch.ones(1, 3, 4)] * 5

        outs = w._collect_artist_outputs(x, ctx, None, {}, artists, "interpolate")

        self.assertEqual(len(outs), 5)
        self.assertEqual(counting.calls, 3)  # chunks of 2 + 2 + 1

    def test_auto_on_cpu_uses_a_single_batched_forward(self):
        counting = _CountingAttn()
        w = CrossAttnWrapper(counting.forward, {"max_batch_artists": 0}, 0)
        x = torch.zeros(1, 2, 4)
        ctx = torch.full((1, 3, 4), 3.0)
        artists = [torch.ones(1, 3, 4)] * 5

        outs = w._collect_artist_outputs(x, ctx, None, {}, artists, "interpolate")

        self.assertEqual(len(outs), 5)
        self.assertEqual(counting.calls, 1)


if __name__ == "__main__":
    unittest.main()
