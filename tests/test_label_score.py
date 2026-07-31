"""Matcher-vs-human scoring functions, hand-computed.

Worked example (10 items):
  matcher preds : 1 1 1 1 1 0 0 0 0 0
  human labels  : 1 1 1 0 0 1 0 0 0 0
  agreements = 7/10  -> po = 0.70
  p_matcher_yes = 0.5, p_human_yes = 0.4
  pe = 0.5*0.4 + 0.5*0.6 = 0.50
  kappa = (0.70 - 0.50) / (1 - 0.50) = 0.40
  confusion: tp=3, fp=2, fn=1, tn=4  -> precision 3/5=0.60, recall 3/4=0.75
"""

import math

from sec_rag.eval.label_score import (
    cohen_kappa,
    confusion,
    kappa_ci,
    precision_recall,
)

_PREDS = [True, True, True, True, True, False, False, False, False, False]
_HUMAN = [True, True, True, False, False, True, False, False, False, False]


def test_confusion_matrix():
    assert confusion(_PREDS, _HUMAN) == {"tp": 3, "fp": 2, "fn": 1, "tn": 4}


def test_precision_recall():
    prec, rec = precision_recall(_PREDS, _HUMAN)
    assert prec == 0.6
    assert rec == 0.75


def test_cohen_kappa_hand_computed():
    assert math.isclose(cohen_kappa(_PREDS, _HUMAN), 0.4, abs_tol=1e-9)


def test_kappa_perfect_and_chance():
    # perfect agreement -> kappa 1.0
    assert cohen_kappa([True, False, True], [True, False, True]) == 1.0
    # matcher always yes, human mixed -> pe from marginals; here po==p_h so kappa 0.
    assert cohen_kappa([True, True, True, True], [True, True, False, False]) == 0.0


def test_precision_recall_undefined_when_no_positives():
    # matcher never fires -> precision undefined (no predicted positives).
    prec, rec = precision_recall([False, False], [True, False])
    assert prec is None
    assert rec == 0.0  # tp=0, fn=1 -> 0/1


def test_kappa_ci_brackets_the_estimate():
    k, lo, hi = kappa_ci(_PREDS, _HUMAN)
    assert math.isclose(k, 0.4, abs_tol=1e-9)
    assert lo < k < hi              # CI brackets the point estimate
    assert hi - lo > 0.3            # n=10 -> wide interval (the honesty point)


def test_empty_is_safe():
    assert cohen_kappa([], []) == 0.0
    assert confusion([], []) == {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
