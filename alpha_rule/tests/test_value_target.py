"""
Tests for the pluggable value-target seam (``mcts.value_target``) that powers
the per-step value target ``z_t``. Replaces the old ``_compute_root_value``.

Pins:
    - ``MaxValue`` = ``node.Q_max`` (best reachable); ``None`` when unvisited.
    - ``ExpectedValue`` = visit-weighted mean over LIVE children of their
      filtered mean (``Q_sum / N_passers``); dead children are EXCLUDED (not
      penalised), so the value target agrees with the policy target.
    - ``MeanPercentileValue`` = the node's own ``Q_sum / N_passers``.
    - ``RealizedReturn`` = ``node.realized_reward`` (``None`` until stamped).
    - ``default_value_target`` auto-pairs with the backup
      (Max -> MaxValue, Percentile -> ExpectedValue).
    - ``run_self_play`` stamps a finite ``state_value`` per step under a
      finite-reward simulator, and stamps ``realized_reward`` on chosen nodes.
"""
from __future__ import annotations

import math

import numpy as np

from alpha_rule.grammar.allen import AllenIntervalGrammar
from alpha_rule.mcts.backprop import MaxRewardBackup, PercentileRewardBackup
from alpha_rule.mcts.expansion import RuleExpansion
from alpha_rule.mcts.self_play import run_self_play
from alpha_rule.mcts.value_target import (
    ExpectedValue,
    MaxValue,
    MeanPercentileValue,
    RealizedReturn,
    default_value_target,
)

_GRAMMAR = AllenIntervalGrammar(event_types=("A", "B", "C"), relations=("<",))
_EXPAND = RuleExpansion(_GRAMMAR)


def _root_with_children():
    root = _GRAMMAR.root()
    a = _EXPAND.expand(root)
    b = _EXPAND.expand(root)
    c = _EXPAND.expand(root)
    return root, a, b, c


# --------------------------------------------------------------------------- #
# MaxValue
# --------------------------------------------------------------------------- #

def test_max_value_is_node_qmax():
    root, *_ = _root_with_children()
    root.N, root.Q_max = 12, 50.0          # backup propagates the subtree max up
    assert MaxValue().state_value(root) == 50.0


def test_max_value_none_when_unvisited():
    root, *_ = _root_with_children()       # root.Q_max == -inf
    assert MaxValue().state_value(root) is None


# --------------------------------------------------------------------------- #
# ExpectedValue — visit-weighted mean of live children's filtered means
# --------------------------------------------------------------------------- #

def test_expected_value_visit_weighted_mean_of_child_filtered_means():
    root, a, b, _c = _root_with_children()
    a.N, a.Q_sum, a.N_passers = 8, 400.0, 8     # filtered mean 50
    b.N, b.Q_sum, b.N_passers = 4, 80.0, 4      # filtered mean 20
    expected = (8 * 50.0 + 4 * 20.0) / (8 + 4)
    assert abs(ExpectedValue().state_value(root) - expected) < 1e-9


def test_expected_value_excludes_dead_children():
    """Point (h): a dead child is dropped from the value target entirely (not
    floored), so it agrees with visit_pi instead of dragging the value down."""
    root, a, b, _c = _root_with_children()
    a.N, a.Q_sum, a.N_passers = 8, 400.0, 8                 # mean 50, live
    b.N, b.Q_sum, b.N_passers, b.is_dead = 4, 80.0, 4, True  # mean 20 but DEAD
    # b excluded -> 50.0, not the (8*50 + 4*20)/12 = 40.0 you'd get if included.
    assert ExpectedValue().state_value(root) == 50.0


def test_expected_value_none_when_no_live_children():
    root, *_ = _root_with_children()       # all children N=0
    assert ExpectedValue().state_value(root) is None


# --------------------------------------------------------------------------- #
# MeanPercentileValue — the node's own filtered mean
# --------------------------------------------------------------------------- #

def test_mean_percentile_value_is_node_filtered_mean():
    root, *_ = _root_with_children()
    root.Q_sum, root.N_passers = 90.0, 3
    assert MeanPercentileValue().state_value(root) == 30.0


def test_mean_percentile_value_none_without_passers():
    root, *_ = _root_with_children()
    assert MeanPercentileValue().state_value(root) is None


# --------------------------------------------------------------------------- #
# RealizedReturn — the node's stamped simulator reward
# --------------------------------------------------------------------------- #

def test_realized_return_reads_node_field():
    _root, a, *_ = _root_with_children()
    assert RealizedReturn().state_value(a) is None     # not stamped yet
    a.realized_reward = 2.5
    assert RealizedReturn().state_value(a) == 2.5


def test_realized_return_none_on_non_finite():
    _root, a, *_ = _root_with_children()
    a.realized_reward = float("-inf")
    assert RealizedReturn().state_value(a) is None


# --------------------------------------------------------------------------- #
# Auto-pairing with the backup operator
# --------------------------------------------------------------------------- #

def test_default_value_target_pairs_with_backup():
    assert isinstance(default_value_target(MaxRewardBackup()), MaxValue)
    assert isinstance(default_value_target(PercentileRewardBackup()), ExpectedValue)


# --------------------------------------------------------------------------- #
# End-to-end: run_self_play records the value target per step.
# --------------------------------------------------------------------------- #

class _ConstantSimulator:
    def __init__(self, value=1.0):
        self.value = value

    def evaluate(self, node):
        return self.value


def test_run_self_play_records_finite_state_value():
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
        assert step.state_value is not None
        assert math.isfinite(step.state_value)


def test_run_self_play_max_value_close_to_constant_reward():
    """Default value target is MaxValue; under a constant-reward simulator the
    root's Q_max saturates to that reward."""
    g = AllenIntervalGrammar(event_types=("A", "B"), relations=("<",))
    traj = run_self_play(
        grammar=g,
        simulator=_ConstantSimulator(value=7.0),
        n_simulations=12,
        depth_limit=2,
        rng=np.random.default_rng(0),
    )
    assert traj.steps[0].state_value is not None
    assert abs(traj.steps[0].state_value - 7.0) < 1.0


def test_run_self_play_realized_return_stamps_chosen_nodes():
    """With RealizedReturn, each non-root step's target is the chosen-step
    reward stamped on the node; the root (never a chosen step) is None."""
    g = AllenIntervalGrammar(event_types=("A", "B"), relations=("<",))
    traj = run_self_play(
        grammar=g,
        simulator=_ConstantSimulator(value=3.0),
        n_simulations=6,
        depth_limit=3,
        value_target=RealizedReturn(),
        rng=np.random.default_rng(0),
    )
    assert traj.steps[0].state_value is None        # root never stamped
    for step in traj.steps[1:]:
        assert step.state_value == 3.0
