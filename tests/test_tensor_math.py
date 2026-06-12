"""Tensor-math and runtime-helper tests (require a real torch install)."""

import os
import sys
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch  # noqa: E402

from anima_mixer import math_utils, patching  # noqa: E402
from anima_mixer.anchor import _context_fingerprint  # noqa: E402
from anima_mixer.wrapper import CrossAttnWrapper, _combine_concat  # noqa: E402


class LowRankTest(unittest.TestCase):
    def test_k_at_least_n_is_identity(self):
        d = torch.randn(3, 16)
        out = math_utils.lowrank_rows_deterministic(d, 3)
        self.assertTrue(torch.equal(out, d))

    def test_projection_reduces_rank(self):
        torch.manual_seed(0)
        d = torch.randn(4, 32)
        out = math_utils.lowrank_rows_deterministic(d, 1)
        # Rank-1 reconstruction: every row is a multiple of the same vector.
        rank = torch.linalg.matrix_rank(out, tol=1e-4).item()
        self.assertEqual(rank, 1)

    def test_projection_is_deterministic(self):
        torch.manual_seed(1)
        d = torch.randn(5, 64)
        a = math_utils.lowrank_rows_deterministic(d, 2)
        b = math_utils.lowrank_rows_deterministic(d.clone(), 2)
        self.assertTrue(torch.allclose(a, b))

    def test_projection_preserves_rowspace_energy_ordering(self):
        torch.manual_seed(2)
        base = torch.randn(1, 16)
        # Rows mostly aligned with one direction plus small noise.
        d = base.repeat(4, 1) + 0.01 * torch.randn(4, 16)
        out = math_utils.lowrank_rows_deterministic(d, 1)
        # The rank-1 reconstruction should be very close to the input.
        self.assertLess((out - d).norm().item() / d.norm().item(), 0.05)


class ProjectPerpendicularTest(unittest.TestCase):
    def test_result_is_orthogonal_to_base_per_token(self):
        torch.manual_seed(3)
        base = torch.randn(2, 5, 8)
        delta = torch.randn(2, 5, 8)
        perp = math_utils.project_perpendicular(delta, base)
        dots = (perp * base).sum(dim=-1)
        self.assertTrue(torch.allclose(dots, torch.zeros_like(dots), atol=1e-5))

    def test_parallel_delta_vanishes(self):
        base = torch.randn(1, 3, 4)
        delta = 2.5 * base
        perp = math_utils.project_perpendicular(delta, base)
        self.assertTrue(torch.allclose(perp, torch.zeros_like(perp), atol=1e-5))


class TimingFadeTest(unittest.TestCase):
    # Window: sigma hi=10 (start), fade_in_lo=8, fade_out_hi=0.5, lo=0.1 (end).
    ROUTE = (0.1, 10.0, 8.0, 0.5)

    def test_outside_window_is_zero(self):
        self.assertEqual(math_utils.timing_fade_factor(self.ROUTE, 11.0), 0.0)
        self.assertEqual(math_utils.timing_fade_factor(self.ROUTE, 0.05), 0.0)

    def test_plateau_is_one(self):
        self.assertEqual(math_utils.timing_fade_factor(self.ROUTE, 5.0), 1.0)

    def test_fade_in_midpoint_is_half(self):
        # sigma 9.0 is halfway between hi=10 and fade_in_lo=8 -> smoothstep(0.5)=0.5
        self.assertAlmostEqual(
            math_utils.timing_fade_factor(self.ROUTE, 9.0), 0.5, places=6,
        )

    def test_fade_out_midpoint_is_half(self):
        # sigma 0.3 is halfway between fade_out_hi=0.5 and lo=0.1
        self.assertAlmostEqual(
            math_utils.timing_fade_factor(self.ROUTE, 0.3), 0.5, places=6,
        )

    def test_no_fade_route_is_binary(self):
        route = (0.1, 10.0, 10.0, 0.1)  # fade edges collapse onto the window
        self.assertEqual(math_utils.timing_fade_factor(route, 10.0), 1.0)
        self.assertEqual(math_utils.timing_fade_factor(route, 0.1), 1.0)
        self.assertEqual(math_utils.timing_fade_factor(route, 10.1), 0.0)

    def test_none_route_or_sigma_is_one(self):
        self.assertEqual(math_utils.timing_fade_factor(None, 5.0), 1.0)
        self.assertEqual(math_utils.timing_fade_factor(self.ROUTE, None), 1.0)


class ResolveMaskTest(unittest.TestCase):
    def test_exact_length_markers(self):
        mask = patching.resolve_mask([0, 1], 2, False, {})
        self.assertEqual(mask, [True, False])

    def test_chunk_expansion_for_batched_latents(self):
        # 2 latents per cond entry: rows [cond, cond, uncond, uncond].
        state = {}
        mask = patching.resolve_mask([0, 1], 4, False, state)
        self.assertEqual(mask, [True, True, False, False])
        self.assertNotIn("_warned", state)

    def test_apply_to_uncond_injects_everywhere(self):
        mask = patching.resolve_mask([0, 1], 4, True, {})
        self.assertEqual(mask, [True] * 4)

    def test_unusable_markers_fall_back_with_warning(self):
        state = {}
        mask = patching.resolve_mask([0, 1], 3, False, state)
        self.assertEqual(mask, [True] * 3)
        self.assertTrue(state.get("_warned"))

    def test_missing_markers_fall_back_with_warning(self):
        state = {}
        mask = patching.resolve_mask(None, 2, False, state)
        self.assertEqual(mask, [True] * 2)
        self.assertTrue(state.get("_warned"))


class BroadcastBatchTest(unittest.TestCase):
    def test_expand_single(self):
        t = torch.randn(1, 4, 8)
        out = patching.broadcast_batch(t, 3)
        self.assertEqual(tuple(out.shape), (3, 4, 8))
        self.assertTrue(torch.equal(out[0], out[2]))

    def test_repeat_divisible(self):
        t = torch.randn(2, 4, 8)
        out = patching.broadcast_batch(t, 4)
        self.assertEqual(tuple(out.shape), (4, 4, 8))
        self.assertTrue(torch.equal(out[0], out[2]))


class _KVMeanAttn(torch.nn.Module):
    """Stub cross-attention: returns the per-batch mean of the K/V tokens,
    broadcast to the query's token count. Lets fusion math be tested without
    a real attention module."""

    def forward(self, x, context=None, rope_emb=None, transformer_options=None):
        mean = context.mean(dim=1, keepdim=True)
        return mean.expand(x.shape[0], x.shape[1], context.shape[-1])


class WrapperHelpersTest(unittest.TestCase):
    def _wrapper(self, state=None):
        return CrossAttnWrapper(torch.nn.Identity(), state or {}, 0)

    def test_combine_concat_scales_and_concatenates(self):
        a = torch.ones(1, 2, 4)
        b = torch.ones(1, 3, 4)
        out = _combine_concat([a, b], [0.5, 2.0])
        self.assertEqual(tuple(out.shape), (1, 5, 4))
        self.assertTrue(torch.allclose(out[0, 0], torch.full((4,), 0.5)))
        self.assertTrue(torch.allclose(out[0, 2], torch.full((4,), 2.0)))

    def test_effective_weights_normalize_then_fade(self):
        # Normalization must run BEFORE the fade multiplies in, otherwise a
        # single fading artist would renormalize back to 1.0 (fade no-op).
        w = self._wrapper({"normalize_weights": True})
        ws, comp = w._effective_weights([1.0], [0.5])
        self.assertEqual(ws, [0.5])
        self.assertAlmostEqual(comp, 0.5)

    def test_effective_weights_no_fade_no_compensation(self):
        w = self._wrapper({"normalize_weights": True})
        ws, comp = w._effective_weights([2.0, 2.0], [1.0, 1.0])
        self.assertEqual(ws, [0.5, 0.5])
        self.assertAlmostEqual(comp, 0.0)

    def test_output_avg_fade_blends_toward_base(self):
        # Single artist at fade 0.5 with normalize on: the output must be the
        # midpoint between the artist attention output and the base output,
        # not a renormalized full-strength artist. match_base_norm is off so
        # the zero-valued stub base output does not rescale the assertion.
        state = {
            "normalize_weights": True, "apply_to_uncond": False,
            "match_base_norm": False,
        }
        w = CrossAttnWrapper(_KVMeanAttn(), state, 0)
        x = torch.zeros(1, 2, 4)
        base_ctx = torch.zeros(1, 3, 4)      # base attention output -> 0
        artist = torch.ones(1, 3, 4)         # artist attention output -> 1
        out = w._fwd_output_avg(
            x, base_ctx, None, {}, [artist], [1.0], [0.5],
            [True], "interpolate", 1.0,
        )
        self.assertTrue(torch.allclose(out, torch.full_like(out, 0.5)))

    def test_artist_chunks_split_by_limit(self):
        w = self._wrapper({"max_batch_artists": 2})
        items = [torch.zeros(1)] * 5
        chunks = w._artist_chunks(items)
        self.assertEqual([len(c) for c in chunks], [2, 2, 1])

    def test_artist_chunks_no_limit(self):
        w = self._wrapper({"max_batch_artists": 0})
        items = [torch.zeros(1)] * 5
        chunks = w._artist_chunks(items)
        self.assertEqual([len(c) for c in chunks], [5])

    def test_apply_fusion_interpolate_respects_mask(self):
        w = self._wrapper()
        base = torch.zeros(2, 3, 4)
        artist = torch.ones(2, 3, 4)
        out = w._apply_fusion(base, artist, [True, False], "interpolate", 0.5)
        self.assertTrue(torch.allclose(out[0], torch.full((3, 4), 0.5)))
        self.assertTrue(torch.allclose(out[1], torch.zeros(3, 4)))

    def test_apply_fusion_base_preserve_keeps_base_direction(self):
        w = self._wrapper()
        base = torch.zeros(1, 2, 4)
        base[..., 0] = 1.0  # base points along e0
        artist = torch.zeros(1, 2, 4)
        artist[..., 0] = 3.0  # parallel to base: should be stripped entirely
        artist[..., 1] = 2.0  # perpendicular: should survive
        out = w._apply_fusion(base, artist, [True], "base_preserve", 1.0)
        self.assertTrue(torch.allclose(out[..., 0], base[..., 0]))
        self.assertTrue(torch.allclose(out[..., 1], torch.full((1, 2), 2.0)))


class MatchBaseNormTest(unittest.TestCase):
    def _wrapper(self):
        return CrossAttnWrapper(torch.nn.Identity(), {}, 0)

    def test_rescales_artist_to_base_rms(self):
        w = self._wrapper()
        base = torch.full((1, 4, 8), 2.0)     # RMS 2
        artist = torch.full((1, 4, 8), 1.0)   # RMS 1
        out = w._match_base_norm(artist, base, [True])
        self.assertTrue(torch.allclose(out, torch.full_like(out, 2.0)))

    def test_scale_is_clamped(self):
        w = self._wrapper()
        base = torch.full((1, 4, 8), 10.0)    # would need 10x
        artist = torch.full((1, 4, 8), 1.0)
        out = w._match_base_norm(artist, base, [True])
        self.assertTrue(torch.allclose(out, torch.full_like(out, 2.0)))  # 2x cap
        base = torch.full((1, 4, 8), 0.1)     # would need 0.1x
        out = w._match_base_norm(artist, base, [True])
        self.assertTrue(torch.allclose(out, torch.full_like(out, 0.5)))  # 0.5x floor

    def test_unmasked_rows_untouched(self):
        w = self._wrapper()
        base = torch.full((2, 4, 8), 2.0)
        artist = torch.full((2, 4, 8), 1.0)
        out = w._match_base_norm(artist, base, [True, False])
        self.assertTrue(torch.allclose(out[0], torch.full((4, 8), 2.0)))
        self.assertTrue(torch.allclose(out[1], torch.full((4, 8), 1.0)))

    def test_preserves_direction(self):
        torch.manual_seed(7)
        w = self._wrapper()
        base = torch.randn(1, 4, 8)
        artist = torch.randn(1, 4, 8)
        out = w._match_base_norm(artist, base, [True])
        cos = torch.nn.functional.cosine_similarity(
            out.flatten(), artist.flatten(), dim=0,
        )
        self.assertGreater(cos.item(), 0.9999)


class AnchorFingerprintTest(unittest.TestCase):
    def test_same_content_same_fingerprint(self):
        a = torch.arange(64, dtype=torch.float32).reshape(1, 8, 8)
        b = a.clone()
        self.assertEqual(_context_fingerprint(a), _context_fingerprint(b))

    def test_different_content_different_fingerprint(self):
        a = torch.zeros(1, 8, 8)
        b = torch.ones(1, 8, 8)
        self.assertNotEqual(_context_fingerprint(a), _context_fingerprint(b))

    def test_none_is_none(self):
        self.assertIsNone(_context_fingerprint(None))


if __name__ == "__main__":
    unittest.main()
