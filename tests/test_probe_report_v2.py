"""Tests for the v27.2 probe report upgrades: contribution shares,
plain-language verdicts, and per-step influence curves.

unittest.TestCase style so both pytest and unittest discovery collect them.
"""

import os
import sys
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from anima_mixer import patching  # noqa: E402
from anima_mixer.nodes_core import (  # noqa: E402
    PROBE_REGISTRY,
    AnimaArtistProbeReport,
    _registry_store,
)
from anima_mixer.probe_stats import (  # noqa: E402
    contribution_shares,
    render_step_curves,
    share_verdict,
)


class ContributionShareTests(unittest.TestCase):
    def test_shares_from_layer_scores(self):
        scores = [[1.0, 1.0], [3.0, 3.0]]  # mean 1.0 vs 3.0
        totals, shares = contribution_shares(scores)
        self.assertEqual(totals, [1.0, 3.0])
        self.assertAlmostEqual(shares[0], 0.25)
        self.assertAlmostEqual(shares[1], 0.75)

    def test_zero_scores_do_not_divide_by_zero(self):
        totals, shares = contribution_shares([[0.0, 0.0], [0.0]])
        self.assertEqual(shares, [0.0, 0.0])

    def test_empty_input(self):
        totals, shares = contribution_shares([])
        self.assertEqual(totals, [])
        self.assertEqual(shares, [])


class ShareVerdictTests(unittest.TestCase):
    def test_bands_are_relative_to_equal_split(self):
        self.assertEqual(share_verdict(0.75, 2), "dominant")      # 1.5x
        self.assertEqual(share_verdict(0.5, 2), "balanced")       # 1.0x
        self.assertEqual(share_verdict(0.2, 3), "balanced")       # 0.6x
        self.assertEqual(share_verdict(0.1, 3), "weak")           # 0.3x
        self.assertEqual(share_verdict(0.02, 3), "negligible")    # 0.06x

    def test_degenerate_counts(self):
        self.assertEqual(share_verdict(1.0, 1), "balanced")
        self.assertEqual(share_verdict(0.5, 0), "")


class StepCurveTests(unittest.TestCase):
    def test_curves_render_in_sampling_order(self):
        step_stats = {
            10.0: [[1.0, 1], [0.5, 1]],
            5.0: [[2.0, 1], [0.5, 1]],
        }
        lines = render_step_curves(step_stats, ["alpha", "beta"])
        text = "\n".join(lines)
        self.assertIn("alpha", text)
        self.assertIn("beta", text)
        # Sampling order is sigma descending: first point sigma 10, last sigma 5.
        self.assertIn("(1.00 -> 2.00)", text)
        self.assertIn("(0.50 -> 0.50)", text)
        self.assertIn("sigma 10 -> 5", text)

    def test_empty_or_missing_stats(self):
        self.assertEqual(render_step_curves(None, ["a"]), [])
        self.assertEqual(render_step_curves({}, ["a"]), [])

    def test_all_zero_stats(self):
        self.assertEqual(render_step_curves({1.0: [[0.0, 1]]}, ["a"]), [])

    def test_missing_artist_rows_default_to_zero(self):
        # A step recorded before an artist list change must not crash.
        lines = render_step_curves({2.0: [[1.0, 1]]}, ["a", "b"])
        self.assertTrue(any("b" in ln for ln in lines))


class ProbeReportV2Tests(unittest.TestCase):
    def _fake_state(self, with_steps=True):
        state = {
            "probe_stats": {
                0: [[2.0, 2], [0.5, 2]],
                1: [[1.0, 2], [0.5, 2]],
            },
            "probe_labels": ["strong_artist", "weak_artist"],
            "probe_num_blocks": 2,
            "_probe_seen_sigmas": {14.0, 7.0},
            "probe_step_stats": {
                14.0: [[1.0, 2], [0.3, 2]],
                7.0: [[0.9, 2], [0.2, 2]],
            },
        }
        if not with_steps:
            state.pop("probe_step_stats")
        return state

    def test_report_contains_shares_verdicts_and_curves(self):
        _registry_store("test-v2", self._fake_state())
        out = AnimaArtistProbeReport().report("test-v2")
        report = out["result"][0]
        self.assertIn("contribution split", report)
        self.assertIn("strong_artist: 75.0%", report)
        self.assertIn("dominant", report)
        self.assertIn("weak_artist: 25.0%", report)
        self.assertIn("balanced", report)
        self.assertIn("per-step influence", report)
        # Legacy sections must survive.
        self.assertIn("relative style influence per layer", report)
        self.assertIn("suggested route", report)

    def test_report_without_step_stats_stays_clean(self):
        # Registry entries written by older probe versions carry no
        # probe_step_stats; the report must skip the curve section.
        PROBE_REGISTRY["legacy-entry"] = {
            "probe_stats": self._fake_state()["probe_stats"],
            "probe_labels": ["strong_artist", "weak_artist"],
            "probe_num_blocks": 2,
            "_probe_seen_sigmas": {14.0},
        }
        out = AnimaArtistProbeReport().report("legacy-entry")
        report = out["result"][0]
        self.assertIn("contribution split", report)
        self.assertNotIn("per-step influence", report)

    def test_registry_store_shares_step_stats_by_reference(self):
        state = self._fake_state()
        _registry_store("ref-check", state)
        state["probe_step_stats"][3.0] = [[0.5, 1], [0.5, 1]]
        self.assertIn(3.0, PROBE_REGISTRY["ref-check"]["probe_step_stats"])


class ResetClearsStepStatsTests(unittest.TestCase):
    def test_reset_clears_step_stats_in_place(self):
        step_stats = {5.0: [[1.0, 1]]}
        state = {
            "probe_stats": {0: [[1.0, 1]]},
            "probe_step_stats": step_stats,
            "_probe_seen_sigmas": {5.0},
        }
        patching.reset_run_state(state)
        self.assertEqual(step_stats, {})  # same dict object, cleared in place
        self.assertEqual(state["probe_step_stats"], {})


class CategoryReorgTests(unittest.TestCase):
    EXPECTED = {
        "AnimaArtistBasic": "Anima/Basic",
        "AnimaArtistStarter": "Anima/Basic",
        "AnimaArtistPack": "Anima/Setup",
        "AnimaArtistPreset": "Anima/Setup",
        "AnimaArtistPresetApply": "Anima/Setup",
        "AnimaArtistSimpleOptions": "Anima/Setup",
        "AnimaArtistOptions": "Anima/Setup",
        "AnimaArtistChainBuilder": "Anima/Setup",
        "AnimaArtistCrossAttn": "Anima/Setup",
        "AnimaArtistInspector": "Anima/Diagnostics",
        "AnimaArtistChainPreview": "Anima/Diagnostics",
        "AnimaArtistProbe": "Anima/Diagnostics",
        "AnimaArtistProbeReport": "Anima/Diagnostics",
        "AnimaArtistTagCheck": "Anima/Diagnostics",
        "AnimaArtistABVariants": "Anima/Diagnostics",
        "AnimaArtistImpactMap": "Anima/Diagnostics",
        "AnimaArtistRecipeSave": "Anima/Recipes",
        "AnimaArtistRecipeLoad": "Anima/Recipes",
    }

    def test_every_node_has_its_new_category(self):
        from anima_mixer import NODE_CLASS_MAPPINGS

        self.assertEqual(set(self.EXPECTED), set(NODE_CLASS_MAPPINGS))
        for name, cls in NODE_CLASS_MAPPINGS.items():
            self.assertEqual(
                getattr(cls, "CATEGORY", None), self.EXPECTED[name], name
            )


if __name__ == "__main__":
    unittest.main()
