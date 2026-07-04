"""Tests for the v27.1 diagnostics nodes (TagCheck / ABVariants / ImpactMap).

One class per node; unittest.TestCase style so both pytest and unittest
discovery collect them.
"""

import os
import sys
import unittest

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from anima_mixer import nodes_diagnostics as diag  # noqa: E402
from anima_mixer.nodes_diagnostics import (  # noqa: E402
    AnimaArtistABVariants,
    AnimaArtistImpactMap,
    AnimaArtistTagCheck,
    build_variants,
    sanitize_label,
)


def _cond(raw):
    """Wrap a raw tensor in the ComfyUI CONDITIONING list format."""
    return [[raw, {}]]


def _unit(dim, idx, size=8):
    v = torch.zeros(size)
    v[idx] = 1.0
    return v


def _raw_from_vector(vec, tokens=4):
    """[1, T, D] tensor whose token-mean pools back to ``vec``."""
    return vec.reshape(1, 1, -1).repeat(1, tokens, 1).clone()


def _pack(base_vec, artist_vecs, labels=None, weights=None, tokens_per_artist=None):
    labels = labels if labels is not None else [f"artist{i}" for i in range(len(artist_vecs))]
    weights = weights if weights is not None else [1.0] * len(artist_vecs)
    conds = []
    for i, vec in enumerate(artist_vecs):
        tokens = tokens_per_artist[i] if tokens_per_artist else 4
        conds.append(_cond(_raw_from_vector(vec, tokens=tokens)))
    return {
        "conditionings": conds,
        "labels": labels,
        "weights": weights,
        "base_conditioning": _cond(_raw_from_vector(base_vec, tokens=6)),
    }


class TagCheckTests(unittest.TestCase):
    def _report(self, pack):
        out = AnimaArtistTagCheck().check(pack)
        self.assertIn("ui", out)
        report = out["result"][0]
        self.assertIsInstance(report, str)
        return report

    def test_identical_to_base_flagged_noop(self):
        base = _unit(8, 0)
        report = self._report(_pack(base, [base.clone()], labels=["ghost"]))
        self.assertIn("[NO-OP] ghost", report)

    def test_distinct_artist_ok(self):
        report = self._report(_pack(_unit(8, 0), [_unit(8, 1)], labels=["wlop"]))
        self.assertIn("[OK] wlop", report)
        self.assertNotIn("[NO-OP] wlop", report)
        self.assertNotIn("[DUPLICATE] wlop", report)
        self.assertNotIn("' and '", report)  # no duplicate-pair line

    def test_near_duplicate_pair_flagged(self):
        a = _unit(8, 1)
        # Same direction, different scale: cosine 1.0 -> duplicate style vector.
        report = self._report(_pack(_unit(8, 0), [a, a * 2.0], labels=["a1", "a2"]))
        self.assertIn("[DUPLICATE]", report)
        self.assertIn("a1", report)
        self.assertIn("a2", report)

    def test_small_shift_is_not_flagged(self):
        # Live calibration (2026-07-04) showed real artists and gibberish
        # overlap in encoder shift; a small-but-nonzero shift must stay [OK].
        base = _unit(8, 0)
        near = base + 0.15 * _unit(8, 1)  # tiny angle from base
        far1, far2 = _unit(8, 2), _unit(8, 3)
        report = self._report(
            _pack(base, [near, far1, far2], labels=["subtle", "strong1", "strong2"])
        )
        self.assertIn("[OK] subtle", report)

    def test_mixed_sequence_lengths_supported(self):
        report = self._report(
            _pack(_unit(8, 0), [_unit(8, 1), _unit(8, 2)], tokens_per_artist=[3, 9])
        )
        self.assertIn("[OK] artist0", report)

    def test_empty_pack_reports_no_artists(self):
        pack = {
            "conditionings": [],
            "labels": [],
            "weights": [],
            "base_conditioning": _cond(_raw_from_vector(_unit(8, 0))),
        }
        report = self._report(pack)
        self.assertIn("no artists", report.lower())

    def test_bad_pack_type_raises(self):
        with self.assertRaises(ValueError):
            AnimaArtistTagCheck().check("not a pack")

    def test_missing_base_conditioning_raises(self):
        pack = _pack(_unit(8, 0), [_unit(8, 1)])
        pack.pop("base_conditioning")
        with self.assertRaises(ValueError):
            AnimaArtistTagCheck().check(pack)

    def test_empty_artist_conditioning_raises(self):
        pack = _pack(_unit(8, 0), [_unit(8, 1)])
        pack["conditionings"][0] = [[None, {}]]
        with self.assertRaises(ValueError):
            AnimaArtistTagCheck().check(pack)


class ABVariantsTests(unittest.TestCase):
    def test_off_vs_full(self):
        chains, labels, report = build_variants("wlop, krenz", "off_vs_full", True, True)
        self.assertEqual(chains, ["", "wlop, krenz"])
        self.assertEqual(len(labels), 2)
        self.assertIn("no_mixer", labels[0])
        self.assertIn("full_mix", labels[1])

    def test_solo_each_with_baselines(self):
        chains, labels, _ = build_variants("wlop, krenz", "solo_each", True, True)
        self.assertEqual(chains, ["", "wlop, krenz", "wlop", "krenz"])
        self.assertIn("solo_wlop", labels[2])
        self.assertIn("solo_krenz", labels[3])

    def test_solo_each_without_baselines(self):
        chains, labels, _ = build_variants("wlop, krenz", "solo_each", False, False)
        self.assertEqual(chains, ["wlop", "krenz"])

    def test_leave_one_out(self):
        chains, labels, _ = build_variants("wlop, krenz, hiten", "leave_one_out", False, True)
        self.assertEqual(
            chains,
            ["wlop, krenz, hiten", "krenz, hiten", "wlop, hiten", "wlop, krenz"],
        )
        self.assertIn("without_wlop", labels[1])

    def test_cumulative_dedupes_full(self):
        chains, labels, report = build_variants("wlop, krenz", "cumulative", True, True)
        # off, +wlop, +krenz(==full) -- full_mix must not appear twice.
        self.assertEqual(chains, ["", "wlop", "wlop, krenz"])
        self.assertEqual(len(chains), len(set(chains)))

    def test_weights_and_comma_routes_preserved(self):
        chain = "1.2::wlop@0,2,4::, krenz@33%-67%"
        chains, labels, _ = build_variants(chain, "solo_each", False, False)
        self.assertEqual(chains, ["1.2::wlop@0,2,4::", "krenz@33%-67%"])

    def test_labels_are_filename_safe_and_unique(self):
        chain = "@yuchi \\(salmon-1000\\), @yuchi \\(salmon-1000\\)@0-8"
        chains, labels, _ = build_variants(chain, "solo_each", True, True)
        self.assertEqual(len(labels), len(set(labels)))
        for label in labels:
            for ch in '\\/:*?"<>|':
                self.assertNotIn(ch, label)

    def test_empty_chain_warns(self):
        chains, labels, report = build_variants("", "solo_each", True, True)
        self.assertEqual(chains, [""])
        self.assertIn("empty", report.lower())

    def test_node_outputs_lists(self):
        node = AnimaArtistABVariants()
        out = node.build("wlop, krenz", "solo_each", True, True)
        chains, labels, report = out["result"]
        self.assertIsInstance(chains, list)
        self.assertIsInstance(labels, list)
        self.assertEqual(len(chains), len(labels))
        self.assertIsInstance(report[0] if isinstance(report, list) else report, str)
        self.assertEqual(AnimaArtistABVariants.OUTPUT_IS_LIST, (True, True, False))

    def test_sanitize_label(self):
        self.assertEqual(sanitize_label('a b\\c:d*e?"f<g>h|i/j'), "a_b_c_d_e_f_g_h_i_j")
        self.assertTrue(sanitize_label("") != "")


class ImpactMapTests(unittest.TestCase):
    def _run(self, a, b, **kwargs):
        node = AnimaArtistImpactMap()
        defaults = {"layout": "triptych", "auto_gain": True, "gain": 4.0}
        defaults.update(kwargs)
        return node.compare(a, b, **defaults)

    def test_identical_images_score_zero(self):
        a = torch.rand(1, 32, 48, 3)
        out = self._run(a, a.clone())
        viz, report, score = out["result"]
        self.assertAlmostEqual(score, 0.0, places=5)
        self.assertEqual(tuple(viz.shape), (1, 32, 48 * 3, 3))

    def test_known_perturbation_score_and_area(self):
        a = torch.full((1, 64, 64, 3), 0.5)
        b = a.clone()
        b[:, :32, :32, :] += 0.2
        out = self._run(a, b)
        _, report, score = out["result"]
        self.assertAlmostEqual(score, 5.0, delta=0.05)  # 0.2 * 25% -> 5.0
        self.assertIn("25.0%", report)

    def test_layouts_shapes(self):
        a, b = torch.rand(1, 16, 24, 3), torch.rand(1, 16, 24, 3)
        for layout, width in (("heatmap", 24), ("overlay", 24), ("triptych", 72)):
            out = self._run(a, b, layout=layout)
            viz = out["result"][0]
            self.assertEqual(tuple(viz.shape), (1, 16, width, 3), layout)
            self.assertGreaterEqual(float(viz.min()), 0.0)
            self.assertLessEqual(float(viz.max()), 1.0)

    def test_batch_broadcast(self):
        a = torch.rand(1, 16, 16, 3)
        b = torch.rand(2, 16, 16, 3)
        out = self._run(a, b, layout="heatmap")
        viz, report, _ = out["result"]
        self.assertEqual(viz.shape[0], 2)

    def test_batch_mismatch_raises(self):
        with self.assertRaises(ValueError):
            self._run(torch.rand(2, 16, 16, 3), torch.rand(3, 16, 16, 3))

    def test_size_mismatch_raises(self):
        with self.assertRaises(ValueError):
            self._run(torch.rand(1, 16, 16, 3), torch.rand(1, 16, 32, 3))

    def test_grayscale_supported(self):
        a, b = torch.rand(1, 16, 16, 1), torch.rand(1, 16, 16, 1)
        out = self._run(a, b, layout="heatmap")
        self.assertEqual(tuple(out["result"][0].shape), (1, 16, 16, 3))

    def test_auto_gain_makes_tiny_diff_visible(self):
        a = torch.full((1, 32, 32, 3), 0.5)
        b = a + 0.002
        out = self._run(a, b, layout="heatmap", auto_gain=True)
        self.assertGreater(float(out["result"][0].max()), 0.5)

    def test_fixed_gain_stays_in_range(self):
        a = torch.zeros(1, 16, 16, 3)
        b = torch.ones(1, 16, 16, 3)
        out = self._run(a, b, layout="overlay", auto_gain=False, gain=50.0)
        viz = out["result"][0]
        self.assertLessEqual(float(viz.max()), 1.0)
        self.assertGreaterEqual(float(viz.min()), 0.0)


class RegistrationTests(unittest.TestCase):
    def test_nodes_registered(self):
        from anima_mixer import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

        for key in ("AnimaArtistTagCheck", "AnimaArtistABVariants", "AnimaArtistImpactMap"):
            self.assertIn(key, NODE_CLASS_MAPPINGS)
            self.assertIn(key, NODE_DISPLAY_NAME_MAPPINGS)

    def test_input_types_shapes(self):
        self.assertIn("artist_pack", AnimaArtistTagCheck.INPUT_TYPES()["required"])
        ab = AnimaArtistABVariants.INPUT_TYPES()["required"]
        self.assertIn("artist_chain", ab)
        self.assertIn("mode", ab)
        imp = AnimaArtistImpactMap.INPUT_TYPES()["required"]
        self.assertIn("image_a", imp)
        self.assertIn("image_b", imp)

    def test_module_has_no_comfy_imports(self):
        # The pack's pytest suite runs without ComfyUI; keep it that way.
        import inspect

        src = inspect.getsource(diag)
        self.assertNotIn("import comfy", src)
        self.assertNotIn("from comfy", src)


if __name__ == "__main__":
    unittest.main()
