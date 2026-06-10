"""
Tests for the MCTS root-value computation that powers
``value_target_mode="root_value"``.

Pins:
    - Visit-weighted average of children's ``Q_max`` is the root value.
    - Dead children (Q_max=-inf) substitute ``reward_floor`` so the
      average stays finite — the value head learns to penalise them.
    - Returns ``None`` when no children have been visited (falls back
      to value_targets per-step fallback rather than propagating
      ``-inf``/``NaN`` to the NN).
    - ``run_self_play`` records a finite root_value per step (when the
      simulator returns finite rewards).
"""
from __future__ import annotations

import math

import numpy as np

from alpha_rule.grammar.allen import AllenIntervalGrammar
from alpha_rule.mcts.expansion import RuleExpansion
from alpha_rule.mcts.replay import DEFAULT_REWARD_FLOOR
from alpha_rule.mcts.self_play import _compute_root_value, run_self_play

_GRAMMAR = AllenIntervalGrammar(event_types=("A", "B", "C"), relations=("<",))
_EXPAND = RuleExpansion(_GRAMMAR)


def _make_root_with_children():
    root = _GRAMMAR.root()
    a = _EXPAND.expand(root)
    b = _EXPAND.expand(root)
    c = _EXPAND.expand(root)
    return root, a, b, c


def test_compute_root_value_visit_weighted_average():
    """Plain visit-weighted average of children's Q_max."""
    root, a, b, c = _make_root_with_children()
    a.N, a.Q_max = 8, 50.0
    b.N, b.Q_max = 4, 30.0
    c.N, c.Q_max = 0, float("-inf")          # unvisited — excluded
    rv = _compute_root_value(root)
    expected = (8 * 50.0 + 4 * 30.0) / (8 + 4)
    assert abs(rv - expected) < 1e-9


def test_compute_root_value_substitutes_floor_for_dead_children():
    """Dead children (Q_max=-inf) contribute reward_floor — pulls the
    average down without producing -inf."""
    root, a, b, _c = _make_root_with_children()
    a.N, a.Q_max = 8, 50.0
    b.N, b.Q_max = 2, float("-inf")          # visited but dead
    rv = _compute_root_value(root)
    expected = (8 * 50.0 + 2 * DEFAULT_REWARD_FLOOR) / (8 + 2)
    assert math.isfinite(rv)
    assert abs(rv - expected) < 1e-9
    # Sanity: with default floor=-100 this is (400 - 200) / 10 = 20.
    assert abs(rv - 20.0) < 1e-9


def test_compute_root_value_returns_none_when_no_visits():
    """No child has been visited yet — return None (callers should
    fall back to a default target rather than propagate to the NN)."""
    root, *_ = _make_root_with_children()
    # All children N=0 by construction.
    assert _compute_root_value(root) is None


def test_compute_root_value_returns_none_when_no_children():
    """Edge case: a node with no children at all."""
    leaf = _GRAMMAR.root()              # fresh node, no children expanded
    assert _compute_root_value(leaf) is None


def test_compute_root_value_custom_dead_penalty():
    """Caller can override the dead penalty (e.g., to align with a
    custom reward_floor on the buffer)."""
    root, a, b, _c = _make_root_with_children()
    a.N, a.Q_max = 6, 60.0
    b.N, b.Q_max = 4, float("-inf")
    rv = _compute_root_value(root, dead_penalty=-50.0)
    expected = (6 * 60.0 + 4 * -50.0) / 10
    assert abs(rv - expected) < 1e-9


# --------------------------------------------------------------------------- #
# End-to-end: run_self_play records root_value on every step.
# --------------------------------------------------------------------------- #


class _ConstantSimulator:
    def __init__(self, value=1.0):
        self.value = value

    def evaluate(self, node):
        return self.value


def test_run_self_play_records_finite_root_value():
    """Every step of a self-play episode under a finite-reward
    simulator carries a finite ``root_value``."""
    g = AllenIntervalGrammar(event_types=("A", "B"), relations=("<",))
    traj = run_self_play(
        grammar=g,
        simulator=_ConstantSimulator(value=5.0),
        n_simulations=8,
        depth_limit=3,
        rng=np.random.default_rng(0),
    )
    assert len(traj.steps) >= 1
    for step in traj.steps:
        assert step.root_value is not None, (
            f"step at state={step.state!r} missing root_value"
        )
        assert math.isfinite(step.root_value), (
            f"non-finite root_value at state={step.state!r}: {step.root_value}"
        )


def test_run_self_play_root_value_close_to_simulator_value():
    """With a constant-reward simulator returning ``value``, the root
    value at each step should be close to that ``value`` (all children's
    Q_max saturate near it)."""
    g = AllenIntervalGrammar(event_types=("A", "B"), relations=("<",))
    target_value = 7.0
    traj = run_self_play(
        grammar=g,
        simulator=_ConstantSimulator(value=target_value),
        n_simulations=12,
        depth_limit=2,
        rng=np.random.default_rng(0),
    )
    # First step's root_value should be close to target_value (all
    # children of the root saw ~target_value during rollouts).
    assert traj.steps[0].root_value is not None
    assert abs(traj.steps[0].root_value - target_value) < 1.0
