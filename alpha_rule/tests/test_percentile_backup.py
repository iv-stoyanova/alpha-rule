"""
Tests for ``mcts.backprop.PercentileRewardBackup``.

Pins:
    - Above-threshold rewards sum into ``Q`` and raise ``Q_max`` along
      the whole leaf-to-root chain. ``N`` increments on every visit.
    - Below-threshold rewards still increment ``N`` on every ancestor
      but leave ``Q`` and ``Q_max`` untouched — this is the deliberate
      departure from ``LegacyNodeBackup``'s orphan-update quirk.
    - During warm-up (fewer than ``min_samples`` past rewards on the
      leaf), the threshold degenerates to the current reward so every
      update is counted as "above threshold".
    - ``-inf`` marks only the leaf dead and never recurses down through
      the tree (regression against the old MaxRewardBackup bug).
    - Dead-parent cascade matches ``MaxRewardBackup``: when every child
      of a fully expanded parent dies, the parent dies and has
      ``Q = Q_max = -inf``.
    - Default constructor uses ``percentile=20, min_samples=10``.

Tree construction uses module-level grammar/expansion singletons (the
node is now pure state; expansion is grammar-driven).
"""
from __future__ import annotations

import math

import numpy as np

from alpha_rule.grammar.allen import AllenIntervalGrammar
from alpha_rule.mcts.backprop import PercentileRewardBackup
from alpha_rule.mcts.expansion import RuleExpansion

_GRAMMAR = AllenIntervalGrammar(event_types=("A", "B"), relations=("<",))
_EXPAND = RuleExpansion(_GRAMMAR)


def _make_root_with_child():
    root = _GRAMMAR.root()
    child = _EXPAND.expand(root)        # first ROOT child = event "A"
    return root, child


# --------------------------------------------------------------------- #
# Case 1: above-threshold updates sum into Q and raise Q_max.
# --------------------------------------------------------------------- #
def test_above_threshold_sums_q_and_tracks_max():
    root, child = _make_root_with_child()
    bp = PercentileRewardBackup(percentile=20, min_samples=10)

    # Seed 10 rewards at 1.0 so the 20th-percentile threshold is 1.0.
    for _ in range(10):
        bp.update(child, 1.0)

    # Post-seed state: every update above (==) threshold, all counted.
    assert child.N == 10
    assert root.N == 10
    assert math.isclose(child.Q, 10.0)
    assert math.isclose(root.Q, 10.0)
    assert child.Q_max == 1.0
    assert root.Q_max == 1.0

    # A clearly above-threshold value: should sum into Q and raise Q_max.
    bp.update(child, 5.0)
    assert child.N == 11
    assert root.N == 11
    assert math.isclose(child.Q, 15.0)
    assert math.isclose(root.Q, 15.0)
    assert child.Q_max == 5.0
    assert root.Q_max == 5.0


# --------------------------------------------------------------------- #
# Case 2: below-threshold still increments N on every ancestor
#         but leaves Q / Q_max untouched. This is the orphan-update fix.
# --------------------------------------------------------------------- #
def test_below_threshold_increments_n_but_not_q():
    root, child = _make_root_with_child()
    bp = PercentileRewardBackup(percentile=20, min_samples=10)

    for _ in range(10):
        bp.update(child, 5.0)            # threshold becomes 5.0
    q_before = child.Q
    qmax_before = child.Q_max
    n_before_child = child.N
    n_before_root = root.N

    bp.update(child, 0.0)                # below-threshold
    assert child.N == n_before_child + 1     # N still climbs
    assert root.N == n_before_root + 1
    assert child.Q == q_before               # Q unchanged
    assert root.Q == q_before
    assert child.Q_max == qmax_before        # Q_max unchanged
    assert root.Q_max == qmax_before


# --------------------------------------------------------------------- #
# Case 3: warmup — fewer than min_samples on the leaf ⇒ threshold
# degenerates to the current value ⇒ update is always counted.
# --------------------------------------------------------------------- #
def test_warmup_treats_every_update_as_above_threshold():
    root, child = _make_root_with_child()
    bp = PercentileRewardBackup(percentile=20, min_samples=10)

    bp.update(child, 0.5)
    bp.update(child, 0.1)
    bp.update(child, 0.9)

    # All three counted as above threshold (because len < min_samples).
    assert child.N == 3
    assert root.N == 3
    assert math.isclose(child.Q, 0.5 + 0.1 + 0.9)
    assert math.isclose(root.Q, 0.5 + 0.1 + 0.9)
    assert child.Q_max == 0.9
    assert root.Q_max == 0.9


# --------------------------------------------------------------------- #
# Case 4: -inf marks ONLY the leaf dead, not ancestors or siblings.
# --------------------------------------------------------------------- #
def test_minus_inf_marks_only_leaf_dead():
    root, child = _make_root_with_child()
    other = _EXPAND.expand(root)              # sibling keeps parent alive

    bp = PercentileRewardBackup(percentile=20, min_samples=10)
    bp.update(child, -np.inf)

    assert child.is_dead is True
    assert other.is_dead is False
    assert root.is_dead is False


# --------------------------------------------------------------------- #
# Case 5: dead cascade — once every child is dead, parent dies too.
# --------------------------------------------------------------------- #
def test_dead_cascade_when_all_children_dead():
    root, child = _make_root_with_child()
    other = _EXPAND.expand(root)

    bp = PercentileRewardBackup(percentile=20, min_samples=10)
    bp.update(child, -np.inf)
    bp.update(other, -np.inf)

    assert child.is_dead and other.is_dead
    assert root.is_dead is True
    assert root.Q == -np.inf
    assert root.Q_max == -np.inf


# --------------------------------------------------------------------- #
# Case 6: default constructor — percentile=20, min_samples=10.
# --------------------------------------------------------------------- #
def test_default_constructor_values():
    bp = PercentileRewardBackup()
    assert bp.percentile == 20
    assert bp.min_samples == 10


# --------------------------------------------------------------------- #
# Case 7: end-to-end — train() accepts the new strategy and runs
# one iteration without raising.
# --------------------------------------------------------------------- #
class _ConstantSimulator:
    def __init__(self, value: float = 1.0):
        self.value = value

    def evaluate(self, node):
        return self.value


def test_train_accepts_percentile_backup_kwarg():
    import pytest

    from alpha_rule.grammar.allen import AllenIntervalGrammar
    # train() is the C6 orchestrator; skip this end-to-end check until it lands.
    train = pytest.importorskip("alpha_rule.training").train

    grammar = AllenIntervalGrammar(event_types=("A", "B"), relations=("<",))
    log = train(
        grammar=grammar,
        expensive_simulator=_ConstantSimulator(value=1.0),
        n_iterations=1,
        n_simulations=4,
        depth_limit=2,
        buffer_capacity=32,
        buffer_warmup=1,
        batch_size=4,
        train_steps_per_iteration=1,
        d_model=16,
        nhead=2,
        num_layers=1,
        max_len=12,
        learning_rate=1e-2,
        seed=0,
        backup=PercentileRewardBackup(percentile=30, min_samples=5),
    )
    assert len(log.iterations) == 1
