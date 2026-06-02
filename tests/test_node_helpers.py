import importlib.util
import sys
import types
import unittest


def load_nodes_module():
    fake_torch = types.ModuleType("torch")
    fake_nn = types.ModuleType("torch.nn")

    class Module:
        pass

    fake_nn.Module = Module
    fake_torch.nn = fake_nn
    fake_torch.inference_mode = lambda: None
    fake_torch.is_tensor = lambda value: False
    sys.modules["torch"] = fake_torch
    sys.modules["torch.nn"] = fake_nn

    spec = importlib.util.spec_from_file_location("nodes_under_test", "nodes.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


nodes = load_nodes_module()


class ArtistRoutingHelpersTest(unittest.TestCase):
    def test_artist_timing_layers_and_weights_parse_together(self):
        parts = nodes._split_artist_chain(
            "::@wlop::1.2@0,2,4%0.0-0.45, "
            "::(krenz:1.1)::0.8@9，18%0.45-0.85, "
            "@hiten"
        )

        parts, timings = nodes._parse_artist_timing_routes(parts)
        parts, layers = nodes._parse_artist_layer_routes(parts)
        names, weights, explicit = nodes._parse_artist_weights(parts)

        self.assertTrue(explicit)
        self.assertEqual(names, ["@wlop", "(krenz:1.1)", "@hiten"])
        self.assertEqual(weights, [1.2, 0.8, 1.0])
        self.assertEqual(layers, ["0,2,4", "9，18", ""])
        self.assertEqual(timings, ["0.0-0.45", "0.45-0.85", ""])

        resolved_layers, has_layers = nodes._resolve_artist_layer_routes(layers, 28)
        self.assertTrue(has_layers)
        self.assertEqual(resolved_layers[0], {0, 2, 4})
        self.assertEqual(resolved_layers[1], {9, 18})

        resolved, has_timings = nodes._resolve_artist_timing_routes(timings)
        self.assertTrue(has_timings)
        self.assertEqual(resolved[0], (0.0, 0.45))
        self.assertEqual(resolved[1], (0.45, 0.85))
        self.assertIsNone(resolved[2])

    def test_compatibility_safe_preset_overrides_risky_settings(self):
        payload = nodes._build_preset_payload(nodes.PRESET_COMPATIBILITY_SAFE)
        self.assertEqual(payload["combine_mode"], nodes.COMBINE_CONCAT)
        self.assertEqual(payload["fusion_mode"], nodes.FUSION_CONCAT_WITH_BASE)
        self.assertTrue(payload["advanced_options"]["compatibility_mode"])

        combine_mode, fusion_mode, _, adv, _ = nodes._merge_runtime_options(
            nodes.COMBINE_OUTPUT_AVG,
            nodes.FUSION_INTERPOLATE,
            1.0,
            {
                "compatibility_mode": True,
                "artist_ema_alpha": 0.5,
                "artist_static_capture": True,
                "artist_anchor_q": True,
            },
            None,
        )

        self.assertEqual(combine_mode, nodes.COMBINE_CONCAT)
        self.assertEqual(fusion_mode, nodes.FUSION_CONCAT_WITH_BASE)
        self.assertEqual(adv["artist_ema_alpha"], 0.0)
        self.assertFalse(adv["artist_static_capture"])
        self.assertFalse(adv["artist_anchor_q"])

        combine_mode, fusion_mode, _, adv, _ = nodes._merge_runtime_options(
            nodes.COMBINE_OUTPUT_AVG,
            nodes.FUSION_INTERPOLATE,
            1.0,
            {"compatibility_mode": False, "artist_static_capture": True},
            payload,
        )
        self.assertEqual(combine_mode, nodes.COMBINE_CONCAT)
        self.assertEqual(fusion_mode, nodes.FUSION_CONCAT_WITH_BASE)
        self.assertTrue(adv["compatibility_mode"])
        self.assertFalse(adv["artist_static_capture"])

    def test_block_map_groups_layers_and_keeps_timing_visible(self):
        block_map = nodes._format_artist_block_map(
            ["wlop", "krenz", "hiten"],
            ["0-1", "2-3", ""],
            ["0.0-0.5", "0.5-1.0", ""],
            num_blocks=4,
            target_blocks=[0, 1, 2, 3],
        )

        self.assertIn("L0-L1: wlop%0.00-0.50, hiten", block_map)
        self.assertIn("L2-L3: krenz%0.50-1.00, hiten", block_map)

    def test_invalid_timing_suffix_stays_in_artist_text(self):
        clean, timing = nodes._parse_artist_timing_route("artist%0.5-0.5")
        self.assertEqual(clean, "artist%0.5-0.5")
        self.assertEqual(timing, "")

    def test_external_cross_attention_wrapper_is_reported(self):
        class PlainCrossAttn:
            context_dim = 1024

        class ExternalWrapper:
            def __init__(self):
                self.original = PlainCrossAttn()

        class Block:
            def __init__(self, cross_attn):
                self.cross_attn = cross_attn

        dm = types.SimpleNamespace(blocks=[Block(PlainCrossAttn()), Block(ExternalWrapper())])
        hints = nodes._describe_external_cross_attn_patches(dm, [0, 1])

        self.assertEqual(len(hints), 1)
        self.assertIn("L1", hints[0])
        self.assertIn("ExternalWrapper", hints[0])

    def test_chain_builder_creates_layer_scheduled_chain(self):
        chain, report = nodes._build_artist_chain_from_rows(
            nodes.CHAIN_LAYOUT_LAYER_SCHEDULED,
            [
                ("@wlop", 1.2, "", ""),
                ("krenz", 0.8, "", ""),
                ("", 1.0, "", ""),
            ],
            num_blocks=28,
        )

        self.assertEqual(
            chain,
            "::@wlop::1.2@0-8%0.0-0.45\n::krenz::0.8@9-18%0.35-0.85",
        )
        self.assertIn("L0-L8: @wlop%0.00-0.45", report)
        self.assertIn("L9-L18: krenz%0.35-0.85", report)

    def test_chain_preview_reports_invalid_timing_before_clip_encoding(self):
        cleaned, report = nodes._format_artist_chain_preview(
            "wlop@0,2,4%0.0-0.5, bad%0.5-0.5",
            num_blocks=28,
        )

        self.assertEqual(cleaned, "wlop@0,2,4%0.0-0.5\nbad%0.5-0.5")
        self.assertIn("L0: wlop%0.00-0.50", report)
        self.assertIn("invalid timing", report)

    def test_chain_builder_ignores_invalid_manual_routes(self):
        chain, report = nodes._build_artist_chain_from_rows(
            nodes.CHAIN_LAYOUT_MANUAL,
            [("wlop", 1.0, "abc", "0.2-0.2")],
            num_blocks=28,
        )

        self.assertEqual(chain, "wlop")
        self.assertIn("invalid layer route ignored", report)
        self.assertIn("invalid timing route ignored", report)

    def test_chain_builder_table_supports_more_than_three_artists(self):
        rows = nodes._parse_builder_artist_table(
            "@a | 1.2\n"
            "b | 0.8\n"
            "c\n"
            "d"
        )
        chain, report = nodes._build_artist_chain_from_rows(
            nodes.CHAIN_LAYOUT_LAYER_SCHEDULED,
            rows,
            num_blocks=28,
        )
        lines = chain.splitlines()

        self.assertEqual(len(lines), 4)
        self.assertEqual(lines[0], "::@a::1.2@0-6%0.00-0.33")
        self.assertEqual(lines[-1], "d@21-27%0.67-1.00")
        self.assertIn("artists: 4", report)

    def test_chain_builder_node_accepts_table_artists(self):
        result = nodes.AnimaArtistChainBuilder().build(
            nodes.CHAIN_LAYOUT_LAYER_SCHEDULED,
            "a\nb\nc\nd\ne",
            "",
            1.0,
            "",
            1.0,
            "",
            1.0,
            num_blocks=28,
        )
        chain, report = result["result"]

        self.assertEqual(len(chain.splitlines()), 5)
        self.assertIn("artists: 5", report)
        self.assertIn("e@22-27%0.72-1.00", chain)


if __name__ == "__main__":
    unittest.main()
