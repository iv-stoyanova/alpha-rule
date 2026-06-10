"""
Tests for the ``Q_sum`` / ``N_passers`` "filtered mean" fields that
cleanly separate "max" from "filtered mean" semantics across the backup
strategies.

The filtered mean of a node is ``Q_sum / N_passers``: the mean of the
values that "counted" under the active backup strategy. ``MaxRewardBackup``
counts every finite sample; ``PercentileRewardBackup`` counts only samples
that clear its percentile threshold.

Also pins ``PUCTSelection(q_source="filtered_mean")`` — the correct PUCT
pairing for ``PercentileRewardBackup`` (it reads ``Q_sum / N_passers``
rather than ``Q_max``).

Tree construction uses module-level grammar/expansion singletons (the
node is now pure state; expansion is grammar-driven).
"""
from __future__ import annotations

import math

import numpy as np

from alpha_rule.grammar.allen import AllenIntervalGrammar
from alpha_rule.mcts.backprop import MaxRewardBackup, PercentileRewardBackup
from alpha_rule.mcts.expansion import RuleExpansion
from alpha_rule.mcts.selection import PUCTSelection

_GRAMMAR = AllenIntervalGrammar(event_types=("A", "B"), relations=("<",))
_EXPAND = RuleExpansion(_GRAMMAR)


def _filtered_mean(node) -> float:
    """The node's filtered mean: mean of the values that counted under
    the active backup strategy. ``0.0`` when nothing has passed yet."""
    return (node.Q_sum / node.N_passers) if node.N_passers > 0 else 0.0


def _leaf():
    """A standalone (parentless) node to accumulate stats on."""
    return _GRAMMAR.root()


# --------------------------------------------------------------------------- #
# MaxRewardBackup maintains Q_sum / N_passers as the mean of all finite samples.
# --------------------------------------------------------------------------- #


def test_max_backup_filtered_mean_counts_every_finite_sample():
    leaf = _leaf()
    backup = MaxRewardBackup()
    for v in [10.0, 50.0, 30.0, 20.0]:
        backup.update(leaf, v)
    # Q_max should be 50; filtered_mean should be the arithmetic mean.
    assert leaf.Q_max == 50.0
    assert leaf.N_passers == 4
    assert abs(leaf.Q_sum - (10.0 + 50.0 + 30.0 + 20.0)) < 1e-9
    assert abs(_filtered_mean(leaf) - 27.5) < 1e-9


def test_max_backup_filtered_mean_skips_minus_inf_samples():
    """A -inf sample marks the leaf dead but should NOT contribute to
    Q_sum / N_passers (would otherwise pollute the filtered mean)."""
    leaf = _leaf()
    backup = MaxRewardBackup()
    backup.update(leaf, 40.0)
    backup.update(leaf, -math.inf)        # death — should not count
    backup.update(leaf, 60.0)
    assert leaf.is_dead is True
    assert leaf.N_passers == 2            # only the finite values
    assert abs(leaf.Q_sum - 100.0) < 1e-9
    assert abs(_filtered_mean(leaf) - 50.0) < 1e-9


# --------------------------------------------------------------------------- #
# PercentileRewardBackup: Q_sum/N_passers counts passers only.
# --------------------------------------------------------------------------- #


def test_percentile_backup_filtered_mean_only_counts_passers():
    """Under PercentileRewardBackup, sub-threshold rewards are excluded
    from Q_sum / N_passers — so the filtered mean is the proper mean
    of passing samples, not diluted by failures."""
    leaf = _leaf()
    backup = PercentileRewardBackup(percentile=20, min_samples=5)

    # Feed a mix of values; small ones will fall below the 20th
    # percentile threshold once we have >= min_samples samples.
    values = [50.0, 60.0, 55.0, 45.0, 5.0, 65.0, 70.0, 1.0, 80.0, 75.0]
    for v in values:
        backup.update(leaf, v)

    assert leaf.N == len(values), "N must count every visit"
    assert leaf.N_passers < leaf.N, "some values should have been filtered"
    # Filtered mean of passers must be HIGHER than the legacy Q/N (which
    # dilutes the numerator over all visits including failures).
    legacy_mean = leaf.Q / leaf.N
    filtered = _filtered_mean(leaf)
    assert filtered > legacy_mean, (
        f"filtered={filtered} should be > legacy={legacy_mean}"
    )


# --------------------------------------------------------------------------- #
# PUCTSelection q_source kwarg
# --------------------------------------------------------------------------- #


def _setup_root_with_two_children():
    root = _GRAMMAR.root()
    a = _EXPAND.expand(root)        # ROOT child "A"
    b = _EXPAND.expand(root)        # ROOT child "B"
    return root, a, b


def test_puct_q_source_max_reads_q_max():
    root, a, _b = _setup_root_with_two_children()
    a.N = 3
    a.Q_max = 50.0
    a.Q_sum = 30.0          # filtered mean would say 10
    a.N_passers = 3
    sel = PUCTSelection(c_puct=0.0, fpu_reduction=0.0, q_source="max")
    assert abs(sel.score(root, a) - 50.0) < 1e-9


def test_puct_q_source_filtered_mean_reads_q_sum_over_n_passers():
    root, a, _b = _setup_root_with_two_children()
    a.N = 5                 # all visits — would dilute legacy Q/N
    a.Q_max = 80.0          # max says 'great'
    a.Q_sum = 60.0          # filtered sum
    a.N_passers = 3         # only 3 of 5 visits passed
    sel = PUCTSelection(c_puct=0.0, fpu_reduction=0.0, q_source="filtered_mean")
    expected = 60.0 / 3.0
    assert abs(sel.score(root, a) - expected) < 1e-9


def test_puct_q_source_filtered_mean_fpu_uses_parent_filtered_mean():
    root, _a, b = _setup_root_with_two_children()
    # Parent has a filtered-mean Q of 25 (and a max of 80 — wrong scale).
    root.Q_sum = 50.0
    root.N_passers = 2
    root.Q_max = 80.0
    sel = PUCTSelection(c_puct=0.0, fpu_reduction=0.0, q_source="filtered_mean")
    # b is unvisited; FPU = parent_filtered_mean = 25.0 (NOT 80).
    assert abs(sel.score(root, b) - 25.0) < 1e-9


def test_puct_q_source_rejects_unknown():
    import pytest
    with pytest.raises(ValueError):
        PUCTSelection(q_source="unknown")
