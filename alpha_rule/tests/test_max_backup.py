"""
Tests for ``mcts.backprop.MaxRewardBackup``.

Pins:
    - Walking from leaf to root, every ancestor's ``Q`` becomes
      ``max(old_Q, value)`` and ``Q_max`` likewise. Every ancestor's
      ``N`` increments by 1.
    - Sub-threshold rewards (which a percentile-filtered backup would
      silently drop) DO update the tree under MaxRewardBackup: every
      finite visit counts.
    - ``-inf`` rewards still mark the leaf dead. Dead-ancestor
      propagation fires only when a parent is fully expanded and all
      its children become dead.
    - Q_max never decreases.

Tree construction uses module-level grammar/expansion singletons (the
node is now pure state; expansion is grammar-driven).
"""
from __future__ import annotations

import numpy as np

from alpha_rule.grammar.allen import AllenIntervalGrammar
from alpha_rule.mcts.backprop import MaxRewardBackup
from alpha_rule.mcts.expansion import RuleExpansion

_GRAMMAR = AllenIntervalGrammar(event_types=("A", "B"), relations=("<",))
_EXPAND = RuleExpansion(_GRAMMAR)


def _make_root_with_child():
    root = _GRAMMAR.root()
    child = _EXPAND.expand(root)        # first ROOT child = event "A"
    return root, child


def test_max_backup_updates_chain_with_max():
    root, child = _make_root_with_child()
    backup = MaxRewardBackup()
    backup.update(child, 0.4)
    backup.update(child, 1.2)
    backup.update(child, 0.7)
    # Both child and root should reflect the maximum (1.2).
    assert child.Q == 1.2
    assert root.Q == 1.2
    assert child.Q_max == 1.2
    assert root.Q_max == 1.2
    assert child.N == 3
    assert root.N == 3


def test_max_backup_q_max_never_decreases():
    root, child = _make_root_with_child()
    backup = MaxRewardBackup()
    for value in [3.0, 1.0, 2.5, 0.5]:
        backup.update(child, value)
    assert child.Q_max == 3.0
    assert root.Q_max == 3.0


def test_max_backup_counts_low_rewards():
    """Max-backup keeps every finite reward: low-but-finite values still
    increment ``N`` along the chain, they just can't lower ``Q``."""
    root2, child2 = _make_root_with_child()
    max_bp = MaxRewardBackup()
    for _ in range(10):
        max_bp.update(child2, 5.0)
    max_bp.update(child2, 0.0)
    assert child2.N == 11             # every visit counted
    assert child2.Q == 5.0            # but Q stays at the running max
    assert root2.N == 11


def test_max_backup_minus_inf_marks_leaf_dead():
    root, child = _make_root_with_child()
    backup = MaxRewardBackup()
    backup.update(child, -np.inf)
    assert child.is_dead is True


def test_max_backup_propagates_dead_when_all_children_dead():
    root, child = _make_root_with_child()
    # Expand the second child too so the parent has a complete sibling set.
    other = _EXPAND.expand(root)
    backup = MaxRewardBackup()
    backup.update(child, -np.inf)
    backup.update(other, -np.inf)
    assert child.is_dead and other.is_dead
    assert root.is_dead is True


def test_max_backup_single_minus_inf_leaf_does_not_kill_root():
    """
    Regression: a SINGLE -inf leaf must NOT mark the root (or any
    ancestor) dead. Only the leaf dies. The tree stays productive for
    subsequent simulations.

    Prior to the fix, ``if value == -np.inf: node.is_dead = True`` lived
    inside the walk-up loop, so one -inf evaluation killed the entire
    root->leaf path in a single call. With sparse-reward grammars that
    meant the first bad sim killed the whole tree.
    """
    root, child = _make_root_with_child()
    # Expand a second child so the parent is NOT "all children dead"
    # after just the first leaf dies.
    other = _EXPAND.expand(root)

    backup = MaxRewardBackup()
    backup.update(child, -np.inf)

    # Leaf dies, root stays alive because `other` is still alive.
    assert child.is_dead is True
    assert other.is_dead is False
    assert root.is_dead is False


def test_max_backup_minus_inf_at_grandchild_only_kills_grandchild():
    """A -inf 2 levels deep must NOT kill root or intermediate parent."""
    root, child = _make_root_with_child()           # root -> child("A")
    other_child = _EXPAND.expand(root)               # root has 2 children

    grandchild = _EXPAND.expand(child)               # child -> "A <END>"
    # Give child a sibling (under the same parent=child) so child isn't
    # fully collapsed the moment its only grandchild dies.
    other_grandchild = _EXPAND.expand(child)         # child -> "A A"

    backup = MaxRewardBackup()
    backup.update(grandchild, -np.inf)

    assert grandchild.is_dead is True
    assert other_grandchild.is_dead is False
    assert child.is_dead is False                    # not all its children dead
    assert other_child.is_dead is False
    assert root.is_dead is False
