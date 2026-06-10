"""
Backprop strategy seam.

A ``BackpropStrategy`` walks from the simulated leaf up to the root, updating
each node's statistics on the way. Two implementations:

    MaxRewardBackup (the default)
        Keeps a running max (``Q_max``). Suited to the spiky reward here --
        almost every rule is junk, a few are good -- so it tracks "is a good
        rule reachable down this branch?" rather than the branch's average.

    PercentileRewardBackup
        Averages only the values above a per-node percentile threshold, so one
        lucky rollout can't peg the score. Use it for a noisy simulator, with
        ``PUCTSelection(q_source="filtered_mean")``.

Both mark a ``-inf`` leaf dead and run the same cascade: a node dies once it
is fully expanded and all its children are dead.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class BackpropStrategy(Protocol):
    """Propagates a value from a leaf up through its ancestors."""

    def update(self, leaf, value: float) -> None: ...


class MaxRewardBackup(BackpropStrategy):
    """
    Maximum backup: keep the best value seen, not the average.

    Walking leaf to root, each node's ``N`` increments and ``Q_max`` takes the
    running max (the diagnostic ``Q`` does too); every finite value also feeds
    the ``Q_sum`` / ``N_passers`` mean. Low-but-finite values still count, they
    just can't lower ``Q_max``. A ``-inf`` value marks the leaf dead.

    A mean backup would drown the rare good rule in the surrounding junk; the
    max keeps it visible, so selection chases branches that can reach a good
    rule rather than branches that are merely good on average.
    """

    def update(self, leaf, value: float) -> None:
        # Mark only the leaf dead on -inf, never the whole path: killing the
        # root-to-leaf chain on one bad rollout would collapse the tree. Death
        # propagates upward only through the cascade below, once every sibling
        # is dead.
        if value == -np.inf:
            leaf.is_dead = True

        # Every finite value counts toward the Q_sum / N_passers mean (no
        # threshold here), read by PUCTSelection(q_source="filtered_mean").
        counts_for_filtered_mean = np.isfinite(value)

        node = leaf
        while node is not None:
            node.N += 1
            if value > node.Q:
                node.Q = value
            if value > node.Q_max:
                node.Q_max = value
            if counts_for_filtered_mean:
                node.Q_sum += float(value)
                node.N_passers += 1

            # Dead-ancestor cascade: once a parent is fully expanded
            # and all its children are dead, the parent dies too.
            parent = node.parent
            if parent is not None and parent.is_fully_expanded() and \
                    parent.children and all(c.is_dead for c in parent.children):
                parent.is_dead = True
                parent.Q = -np.inf
                parent.Q_max = -np.inf

            node = parent


class PercentileRewardBackup(BackpropStrategy):
    """
    Percentile-filtered mean backup, for a noisy simulator where
    ``MaxRewardBackup`` would pin ``Q_max`` to one lucky rollout.

    The threshold is the ``percentile`` of the leaf's own ``past_rewards``
    history (during warm-up, ``len(past_rewards) < min_samples``, it is the
    current value so the update always counts). On each ``update(leaf, value)``:

    1. ``value == -inf`` marks only the leaf dead (not the whole chain).
    2. A finite ``value`` is appended to ``leaf.past_rewards``.
    3. Walking leaf to root, every node's ``N`` increments (so PUCT always has
       accurate visit counts). If ``value`` is finite and ``>= threshold`` it
       also updates ``Q``, ``Q_max`` and the ``Q_sum`` / ``N_passers`` mean
       read by ``PUCTSelection(q_source="filtered_mean")``.
    4. The same dead cascade as ``MaxRewardBackup`` runs.
    """

    def __init__(self, percentile: float = 20, min_samples: int = 10):
        self.percentile = percentile
        self.min_samples = min_samples

    def update(self, leaf, value: float) -> None:
        if value == -np.inf:
            leaf.is_dead = True

        if np.isfinite(value):
            leaf.past_rewards.append(value)

        if len(leaf.past_rewards) >= self.min_samples:
            threshold = np.percentile(leaf.past_rewards, self.percentile)
        else:
            threshold = value

        counts_for_q = np.isfinite(value) and value >= threshold

        node = leaf
        while node is not None:
            node.N += 1
            if counts_for_q:
                node.Q += value
                if value > node.Q_max:
                    node.Q_max = value
                # Q_sum / N_passers is the mean of the values that passed the
                # threshold -- the percentile mean read by selection, distinct
                # from Q/N which would dilute over all visits.
                node.Q_sum += float(value)
                node.N_passers += 1

            parent = node.parent
            if parent is not None and parent.is_fully_expanded() and \
                    parent.children and all(c.is_dead for c in parent.children):
                parent.is_dead = True
                parent.Q = -np.inf
                parent.Q_max = -np.inf

            node = parent
