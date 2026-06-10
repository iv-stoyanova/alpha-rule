"""
Backpropagation strategy seam.

A ``BackpropStrategy`` walks from the simulated leaf up toward the root,
updating statistics on every node it touches.

Two implementations:

    - ``MaxRewardBackup`` (the self-play default): AlphaZero-style maximum
      backup. Walks the chain setting ``Q = max(Q, value)`` and
      ``Q_max = max(Q_max, value)``; every finite value also feeds the
      ``Q_sum`` / ``N_passers`` filtered mean. Best for the spiky symbolic
      reward landscape, it answers "does a high-reward formula exist down
      this branch?".

    - ``PercentileRewardBackup``: percentile-filtered sum-then-mean backup
      for noisy simulators where a single lucky rollout would otherwise peg
      ``Q_max``. Pair with ``PUCTSelection(q_source="filtered_mean")``.

Both mark a ``-inf`` leaf dead and run the same dead-ancestor cascade:
a parent dies once it is fully expanded and all its children are dead.
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
    AlphaZero-style maximum backpropagation.

    Walk from leaf to root setting ``Q = max(Q, value)`` and ``Q_max``
    likewise. Every visited ancestor's ``N`` increments by 1 (so
    PUCT's exploration term still has visit counts to work with).

    No percentile filter, no orphan-drop: low-but-finite rewards still
    update the chain, they just can't lower ``Q``.

    ``-inf`` rewards mark the leaf dead and trigger the
    "all siblings dead then parent dead" cascade shared with
    ``PercentileRewardBackup``.

    Why this matters for symbolic search: the reward landscape over
    formulas is spiky, almost everything is junk, only a handful of
    derivations are good. A mean backup drowns the rare successes in
    surrounding garbage; max-backup answers "does a high-reward formula
    exist down this branch?", which is exactly what MCTS should optimise
    for here.
    """

    def update(self, leaf, value: float) -> None:
        # Mark ONLY the leaf dead on -inf, not every ancestor. Killing
        # the whole root-to-leaf path on a single bad simulation would make
        # sparse-reward grammars collapse the tree (later self-play
        # iterations then produce empty trajectories). Upward death is
        # handled by the dead-ancestor cascade below, which fires only
        # when every sibling at a level is dead.
        if value == -np.inf:
            leaf.is_dead = True

        # Every finite value counts for the filtered mean (no
        # thresholding here); Q_sum / N_passers feed
        # PUCTSelection(q_source="filtered_mean").
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
    Percentile-filtered mean backup.

    ``MaxRewardBackup`` can be optimistic on noisy simulators (a single
    lucky rollout pins ``Q_max`` forever). This strategy keeps a
    mean-style aggregation but only over samples that clear a percentile
    threshold.

    Semantics
    ---------
    Percentile is a *strategy-level* config (constructor arg), not a
    per-node field. The threshold is computed from the leaf's own
    ``past_rewards`` history. During warm-up
    (``len(past_rewards) < min_samples``) the threshold degenerates to
    the current value so the update always counts.

    On each call to ``update(leaf, value)``:

    1. ``value == -inf`` marks **only the leaf** dead (same as
       ``MaxRewardBackup``, never nukes the whole chain).
    2. Finite samples are appended to ``leaf.past_rewards``.
    3. The ancestor chain is walked root-ward. For every visited node:
       - ``N`` increments **always**, so PUCT's exploration term sees
         accurate visit counts even when this value is below threshold.
       - If ``value`` is finite and ``value >= threshold``:
         ``Q += value``, ``Q_max = max(Q_max, value)``, and the value
         joins the ``Q_sum`` / ``N_passers`` filtered mean (the proper
         percentile mean, read by ``PUCTSelection(q_source="filtered_mean")``).
    4. The same dead-ancestor cascade as ``MaxRewardBackup`` runs.

    vs. ``MaxRewardBackup``: aggregates as a mean of passing samples
    instead of a running max, so noisy simulators don't peg the value to
    one lucky sample.
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
                # The "filtered mean" under percentile-backup is the
                # mean of *passing* values: Q_sum/N_passers updates
                # only when the value passed the threshold. This is
                # the correct percentile mean, distinct from Q/N
                # (which dilutes the numerator over all visits).
                node.Q_sum += float(value)
                node.N_passers += 1

            parent = node.parent
            if parent is not None and parent.is_fully_expanded() and \
                    parent.children and all(c.is_dead for c in parent.children):
                parent.is_dead = True
                parent.Q = -np.inf
                parent.Q_max = -np.inf

            node = parent
