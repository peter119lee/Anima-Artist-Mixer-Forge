"""Test v26 syntax enhancements: prefix weight and base_prompt expansion."""

import pytest
from anima_mixer.parsing import parse_artist_weights, expand_prompt_weights, split_artist_chain


class TestPrefixWeightSyntax:
    """Test v26 prefix weight syntax: weight::name"""

    def test_prefix_syntax_basic(self):
        parts = ["1.5::wlop"]
        names, weights, has_explicit = parse_artist_weights(parts)
        assert names == ["wlop"]
        assert weights == [1.5]
        assert has_explicit is True

    def test_prefix_with_parentheses(self):
        parts = ["0.8::(wlop:1.1)"]
        names, weights, has_explicit = parse_artist_weights(parts)
        assert names == ["(wlop:1.1)"]
        assert weights == [0.8]
        assert has_explicit is True

    def test_postfix_still_works(self):
        """Backward compatibility: postfix syntax still works"""
        parts = ["wlop::1.5", "::sakimichan::2.0"]
        names, weights, has_explicit = parse_artist_weights(parts)
        assert names == ["wlop", "sakimichan"]
        assert weights == [1.5, 2.0]
        assert has_explicit is True

    def test_mixed_prefix_and_postfix(self):
        """Both syntaxes can coexist in same chain"""
        parts = ["1.5::wlop", "krenz::0.8"]
        names, weights, has_explicit = parse_artist_weights(parts)
        assert names == ["wlop", "krenz"]
        assert weights == [1.5, 0.8]
        assert has_explicit is True

    def test_decorative_prefix(self):
        """::wlop without weight should work (decorative)"""
        parts = ["::wlop"]
        names, weights, has_explicit = parse_artist_weights(parts)
        assert names == ["wlop"]
        assert weights == [1.0]
        assert has_explicit is False

    def test_no_weight(self):
        """Plain name without weight"""
        parts = ["wlop"]
        names, weights, has_explicit = parse_artist_weights(parts)
        assert names == ["wlop"]
        assert weights == [1.0]
        assert has_explicit is False

    def test_weight_clamping(self):
        """Weights should be clamped to [0.0, 4.0]"""
        parts = ["5.0::wlop", "-2.0::krenz"]
        names, weights, has_explicit = parse_artist_weights(parts)
        assert names == ["wlop", "krenz"]
        assert weights == [4.0, -2.0]  # 5.0 clamped to 4.0, -2.0 is valid
        assert has_explicit is True

    def test_invalid_weight_keeps_raw(self):
        """Invalid weight should keep raw text"""
        parts = ["abc::wlop"]
        names, weights, has_explicit = parse_artist_weights(parts)
        assert names == ["abc::wlop"]  # Kept as-is
        assert weights == [1.0]
        assert has_explicit is False


class TestBasePromptWeightExpansion:
    """Test v26 base_prompt weight syntax: weight::target::"""

    def test_basic_expansion(self):
        text = "1.5::masterpiece::, 1girl"
        result = expand_prompt_weights(text)
        assert result == "(masterpiece:1.5), 1girl"

    def test_multi_word_target(self):
        text = "1.3::detailed background, intricate::, 1girl"
        result = expand_prompt_weights(text)
        assert result == "(detailed background, intricate:1.3), 1girl"

    def test_middle_of_prompt(self):
        text = "masterpiece, 1.5::high quality::, ok"
        result = expand_prompt_weights(text)
        assert result == "masterpiece,(high quality:1.5), ok"  # Space before comma is consumed

    def test_multiple_expansions(self):
        text = "1.5::masterpiece::, 1.2::high quality::, 1girl"
        result = expand_prompt_weights(text)
        assert result == "(masterpiece:1.5),(high quality:1.2), 1girl"  # Space before comma is consumed

    def test_missing_trailing_delimiter(self):
        """Missing trailing :: should keep original"""
        text = "1.5::masterpiece, 1girl"
        result = expand_prompt_weights(text)
        assert result == "1.5::masterpiece, 1girl"  # Unchanged

    def test_invalid_weight(self):
        """Non-numeric weight should keep original"""
        text = "abc::masterpiece::, 1girl"
        result = expand_prompt_weights(text)
        assert result == "abc::masterpiece::, 1girl"  # Unchanged

    def test_no_double_colon(self):
        """Text without :: should pass through unchanged"""
        text = "masterpiece, 1girl"
        result = expand_prompt_weights(text)
        assert result == "masterpiece, 1girl"

    def test_weight_clamping(self):
        """Weights should be clamped to [0.0, 4.0]"""
        text = "5.0::masterpiece::, 1girl"
        result = expand_prompt_weights(text)
        assert result == "(masterpiece:4), 1girl"  # 5.0 clamped to 4.0

    def test_preserves_comfy_brackets(self):
        """Should not touch existing (name:1.5) syntax"""
        text = "(masterpiece:1.5), 1girl"
        result = expand_prompt_weights(text)
        assert result == "(masterpiece:1.5), 1girl"  # Unchanged

    def test_newline_boundary(self):
        """Should not cross newlines"""
        text = "1.5::masterpiece\n, 1girl"
        result = expand_prompt_weights(text)
        # Should not match because target crosses newline
        assert "::" in result  # Original preserved


class TestV26Integration:
    """Integration tests for v26 features"""

    def test_full_chain_with_prefix_syntax(self):
        chain = "1.5::wlop, 0.8::sakimichan, krenz"
        parts = split_artist_chain(chain)
        names, weights, has_explicit = parse_artist_weights(parts)

        assert len(names) == 3
        assert names == ["wlop", "sakimichan", "krenz"]
        assert weights == [1.5, 0.8, 1.0]
        assert has_explicit is True

    def test_base_prompt_with_expansion(self):
        base = "1.5::masterpiece::, 1.2::high quality::, 1girl"
        expanded = expand_prompt_weights(base)

        assert expanded == "(masterpiece:1.5),(high quality:1.2), 1girl"  # Space before comma is consumed
        assert "(" in expanded and ")" in expanded
