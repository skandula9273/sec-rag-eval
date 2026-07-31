"""Numeric normalizer + matcher for answer accuracy (pure, no network).

Hand-computed expected values, including the adversarial cases the whole scorer
hinges on: scale mismatch, negatives (parens + minus), percentages, and a gold
that is a sentence rather than a figure.
"""

import pytest

from sec_rag.eval.answer_accuracy import (
    extract_numbers,
    is_refusal,
    numeric_match,
)


@pytest.mark.parametrize(
    "text,value,is_pct",
    [
        ("$1.2B", 1.2e9, False),           # single-letter scale
        ("1,200 million", 1.2e9, False),   # word scale + thousands separator
        ("$1,577.00", 1577.0, False),      # commas + decimal, no scale (bare)
        ("$2,018mn", 2.018e9, False),      # 'mn' suffix
        ("$59,268 million", 5.9268e10, False),
        ("(1,234)", -1234.0, False),       # accounting parens -> negative
        ("-$1,561M", -1.561e9, False),     # leading minus + scale
        ("16%", 16.0, True),               # percent kept as its own kind
        ("101.5 percent", 101.5, True),
        ("0", 0.0, False),
        ("24.26", 24.26, False),           # bare ratio
    ],
)
def test_extract_first_number(text, value, is_pct):
    got = extract_numbers(text)[0]
    assert got == (value, is_pct)


def test_extract_strips_citation_markers():
    # "[1]" / "[2]" are references, not figures — must not become 1, 2.
    assert extract_numbers("total was $5,000 million [1][2]") == [(5.0e9, False)]


def test_scale_mismatch_matches():
    # The headline requirement: $1.2B and "1,200 million" are the same value.
    assert numeric_match("$1.2B", "Revenue was 1,200 million.") is True
    # Bare-millions convention: gold stores 59268, answer says "$59,268 million".
    assert numeric_match("$59268.00", "Total assets were $59,268 million [1].") is True
    # A genuine 1000x-off wrong scale that ISN'T a power-of-1000 bridge stays wrong.
    assert numeric_match("24.26", "the ratio was 2426") is False


def test_negatives_and_sign():
    assert numeric_match("-$1,561M", "Working capital was -$1,561 million.") is True
    assert numeric_match("(1,234)", "Operating loss of $(1,234) million was...") is True
    # sign must match: negative gold vs positive candidate of the same magnitude.
    assert numeric_match("-1561", "the value was 1561") is False


def test_percentages():
    assert numeric_match("101.5%", "growth rate = 101.5%") is True
    assert numeric_match("1.9%", "capex/revenue was 1.90 percent") is True  # abs tol
    # a percent gold must never scale-bridge to a raw number.
    assert numeric_match("1.9%", "the figure was 1900") is False
    # percent gold vs a different percent -> wrong.
    assert numeric_match("36.8%", "operating margin was 34.6%") is False


def test_zero():
    assert numeric_match("0", "Restructuring costs were 0 for the period.") is True
    assert numeric_match("0", "Costs were $5 million.") is False


def test_sentence_gold_is_not_applicable():
    # A sentence with no figure -> numeric matcher abstains (None), defers to the LLM.
    assert numeric_match("There are none.", "The company has no such securities.") is None
    assert numeric_match(
        "Amcor entered into supplemental indentures relating to its notes.",
        "Amcor filed supplemental indentures.",
    ) is None
    # A multi-number sentence gold is also ambiguous -> None (not a single figure).
    assert numeric_match("declined from 36.8% in 2021 to 34.6% in 2022", "…") is None


def test_rounding_tolerance():
    # ~1% rounding is accepted; a clearly different number is not.
    assert numeric_match("101.5%", "the rate was 101.47%") is True
    assert numeric_match("$1577.00", "capex was $1,580 million") is True   # 0.19% off
    assert numeric_match("$1577.00", "capex was $1,700 million") is False  # 7.8% off


def test_is_refusal():
    assert is_refusal("I cannot answer this from the provided sources.") is True
    assert is_refusal("The sources do not contain FY2022 current assets.") is True
    # a real answer that merely contains 'cannot' is NOT a refusal.
    assert is_refusal("3M cannot sustain its dividend without new debt.") is False
    assert is_refusal("Revenue was $5 billion [1].") is False
