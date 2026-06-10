"""
Tests for ``mcts.selection.PUCTSelection``.

Pins:
    - ``score`` matches the AlphaZero formula exactly given hand-set
      ``child.N``, ``child.Q_max``, ``child.prior``, ``parent.N``.
    - Unvisited children get ``Q = 0`` (FPU); the exploration term
      dominates so they get explored first when priors are equal.
    - Dead children score ``-inf`` and are never selected as long as a
      live sibling exists.
    - ``select`` returns the highest-scoring non-dead child, or None
      if all children are dead.
"""
from __future__ import annotations

import math

from alpha_rule.grammar.allen import AllenIntervalGrammar
from alpha_rule.mcts.expansion import RuleExpansion
from alpha_rule.mcts.selection import PUCTSelection

_GRAMMAR = AllenIntervalGrammar(event_types=("A", "B"), relations=("<",))
_EXPAND = RuleExpansion(_GRAMMAR)


def _setup_two_children(visited_q_max=0.5, visited_n=10, visited_prior=0.4,
                         unvisited_prior=0.6):
    """Build a root with two children: one visited, one unvisited."""
    root = _GRAMMAR.root()
    a = _EXPAND.expand(root)            # "A"
    b = _EXPAND.expand(root)            # "B"
    a.N = visited_n
    a.Q_max = visited_q_max
    a.prior = visited_prior
    b.N = 0
    b.Q_max = float("-inf")
    b.prior = unvisited_prior
    root.N = visited_n
    return root, a, b


def test_score_matches_puct_formula_with_fpu_reduction_zero():
    """With ``fpu_reduction=0`` and a parent whose Q_max is unset, the
    unvisited child's FPU collapses to 0 — matches the historical
    AlphaGo-Zero behaviour."""
    root, a, b = _setup_two_children(
        visited_q_max=0.5, visited_n=4, visited_prior=0.3, unvisited_prior=0.7,
    )
    root.N = 4
    sel = PUCTSelection(c_puct=1.5, fpu_reduction=0.0)
    sum_n = 4                                     # max(1, sum of children's N)
    expected_a = 0.5 + 1.5 * 0.3 * math.sqrt(sum_n) / (1 + 4)
    expected_b = 0.0 + 1.5 * 0.7 * math.sqrt(sum_n) / (1 + 0)
    assert abs(sel.score(root, a) - expected_a) < 1e-9
    assert abs(sel.score(root, b) - expected_b) < 1e-9


def test_score_default_fpu_reduces_unvisited_child_q():
    """With the default ``fpu_reduction=0.25``, an unvisited child's q
    drops by ``0.25 * sqrt(visited_prior_mass)``."""
    root, a, b = _setup_two_children(
        visited_q_max=0.5, visited_n=4, visited_prior=0.3, unvisited_prior=0.7,
    )
    root.N = 4
    sel = PUCTSelection(c_puct=1.5)               # default fpu_reduction=0.25
    sum_n = 4
    # Parent's Q_max is -inf -> parent_q falls back to 0.0.
    visited_prior_mass = 0.3
    fpu = 0.0 - 0.25 * math.sqrt(visited_prior_mass)
    expected_b = fpu + 1.5 * 0.7 * math.sqrt(sum_n) / (1 + 0)
    assert abs(sel.score(root, b) - expected_b) < 1e-9


def test_unvisited_child_with_higher_prior_wins_first():
    root, _, b = _setup_two_children(
        visited_q_max=0.1, visited_n=2, visited_prior=0.4, unvisited_prior=0.6,
    )
    sel = PUCTSelection(c_puct=1.5)
    chosen = sel.select(root)
    assert chosen is b


def test_dead_child_scores_minus_inf_and_never_picked():
    root, a, b = _setup_two_children()
    a.is_dead = True
    sel = PUCTSelection()
    assert sel.score(root, a) == float("-inf")
    chosen = sel.select(root)
    assert chosen is b


def test_select_returns_none_when_all_children_dead():
    root, a, b = _setup_two_children()
    a.is_dead = True
    b.is_dead = True
    sel = PUCTSelection()
    assert sel.select(root) is None


def test_score_uses_q_max_not_mean():
    """``Q_max`` is the convention; mean is irrelevant under MaxRewardBackup."""
    root, a, _ = _setup_two_children(visited_q_max=10.0, visited_n=5)
    a.Q = -5.0                                     # mean would say 'bad'
    a.Q_max = 10.0                                 # max says 'great'
    sel = PUCTSelection(c_puct=0.0)                # no exploration
    score = sel.score(root, a)
    assert abs(score - 10.0) < 1e-9


# --------------------------------------------------------------------------- #
# FPU (first-play urgency) — KataGo-style fix.
# --------------------------------------------------------------------------- #


def test_fpu_uses_parent_q_max_when_known():
    """When the parent has a known Q_max and ``fpu_reduction=0``, an
    unvisited child's exploitation term equals the parent's Q_max."""
    root, _, b = _setup_two_children()
    root.Q_max = 50.0                              # parent has a known Q
    sel = PUCTSelection(c_puct=0.0, fpu_reduction=0.0)
    assert abs(sel.score(root, b) - 50.0) < 1e-9


def test_fpu_applies_reduction_with_visited_prior_mass():
    """FPU = parent.Q_max - fpu_reduction * sqrt(visited_prior_mass).
    Confirm the reduction term depends on the *visited* siblings' prior."""
    root, a, b = _setup_two_children(
        visited_q_max=42.0, visited_n=3,
        visited_prior=0.4, unvisited_prior=0.6,
    )
    root.Q_max = 100.0
    sel = PUCTSelection(c_puct=0.0, fpu_reduction=0.5)
    # visited_prior_mass = sum(c.prior for visited children) = 0.4 (only A).
    expected_fpu = 100.0 - 0.5 * math.sqrt(0.4)
    assert abs(sel.score(root, b) - expected_fpu) < 1e-6


def test_fpu_negative_reduction_rejected():
    import pytest
    with pytest.raises(ValueError):
        PUCTSelection(fpu_reduction=-0.1)


def test_fpu_unvisited_no_longer_strictly_worse_than_positive_visited():
    """In a positive-reward domain, the old FPU=0 made unvisited
    children strictly worse than any visited sibling with Q_max>0.
    With KataGo-style FPU and a parent whose Q_max matches the visited
    child's Q_max, the unvisited child is competitive."""
    root, a, b = _setup_two_children(
        visited_q_max=50.0, visited_n=10,
        visited_prior=0.4, unvisited_prior=0.4,
    )
    root.Q_max = 50.0                              # parent's Q matches visited child's
    sel = PUCTSelection(c_puct=1.5, fpu_reduction=0.25)
    score_a = sel.score(root, a)
    score_b = sel.score(root, b)
    # b (unvisited) gets a fresher exploration term — the visit-count
    # denominator (1+10 vs 1+0) makes its u-term dominate. With FPU = 0
    # (old behaviour) b would have been even more dominant; here we just
    # assert b is competitive (within 50% of a's score, not strictly less
    # by orders of magnitude).
    assert score_b > 0.5 * score_a, (
        f"unvisited child unexpectedly suppressed: a={score_a}, b={score_b}"
    )
