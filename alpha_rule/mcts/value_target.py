"""
Value-target strategy seam.

A ``ValueTarget`` turns the MCTS statistics at a state ``s_t`` into the scalar
``z_t`` the network's value head regresses. It is pluggable like the
backup/selection seams so the value target can MATCH the search operator
(otherwise the value head is trained on a different quantity than the search
optimises). Four implementations:

    MaxValue            z = node.Q_max
                        "best rule reachable from here". Pairs with
                        ``MaxRewardBackup`` (the default) and with
                        ``PUCTSelection(q_source="max")``.

    ExpectedValue       z = visit-weighted mean over LIVE children of each
                        child's filtered mean (``Q_sum / N_passers``)
                        "value of the current tree policy". Pairs with
                        ``PercentileRewardBackup``.

    MeanPercentileValue z = node.Q_sum / node.N_passers
                        the node's OWN filtered mean -- the same statistic
                        ``PUCTSelection(q_source="filtered_mean")`` reads for
                        selection.

    RealizedReturn      z = node.realized_reward
                        the honest ``simulator.evaluate`` of this state's rule,
                        stored on the node when it was the chosen step during
                        self-play. A ground-truth, on-policy target rather than
                        a search-derived one. ``None`` for states never
                        evaluated as a chosen step (e.g. the episode root).

Dead branches: every tree-derived target ignores dead children for free -- a
dead branch's ``-inf`` Q_max never wins the ``MaxValue`` max, and the
``ExpectedValue`` mean iterates only live children -- so there is no
dead-branch penalty. This keeps the value target and the policy target
(``_normalised_visit_distribution``, which also skips dead children) in
agreement about which branches exist.

``state_value`` returns ``None`` when no usable statistic exists yet; callers
fall back to a finite per-step default rather than feed ``None`` / ``-inf``
into the value loss.
"""
from __future__ import annotations

import math
from typing import Optional, Protocol, runtime_checkable

from alpha_rule.mcts.backprop import PercentileRewardBackup


@runtime_checkable
class ValueTarget(Protocol):
    """Maps an MCTS node to the value target ``z_t`` for its state."""

    def state_value(self, node) -> Optional[float]: ...


class MaxValue:
    """``z = node.Q_max`` -- best rule reachable from this state."""

    def state_value(self, node) -> Optional[float]:
        q = node.Q_max
        return float(q) if math.isfinite(q) else None


class ExpectedValue:
    """``z`` = visit-weighted mean over live children of their filtered mean
    (``Q_sum / N_passers``) -- the value of the current tree policy."""

    def state_value(self, node) -> Optional[float]:
        num = 0.0
        den = 0
        for c in node.children:
            if c.is_dead or c.N <= 0 or c.N_passers <= 0:
                continue
            num += c.N * (c.Q_sum / c.N_passers)
            den += c.N
        return (num / den) if den > 0 else None


class MeanPercentileValue:
    """``z = node.Q_sum / node.N_passers`` -- the node's own filtered mean,
    the same statistic ``PUCTSelection(q_source="filtered_mean")`` reads."""

    def state_value(self, node) -> Optional[float]:
        return (node.Q_sum / node.N_passers) if node.N_passers > 0 else None


class RealizedReturn:
    """``z = node.realized_reward`` -- the simulator's own evaluation of this
    state's rule, stamped on the node when it was the chosen step during
    self-play. A ground-truth, on-policy target. ``None`` for states never
    evaluated as a chosen step (e.g. the episode root)."""

    def state_value(self, node) -> Optional[float]:
        r = getattr(node, "realized_reward", None)
        return float(r) if (r is not None and math.isfinite(r)) else None


def default_value_target(backup) -> ValueTarget:
    """Pick the value target that matches the backup operator so the value
    head regresses the same quantity the search optimises:

        PercentileRewardBackup -> ExpectedValue
        anything else (incl. MaxRewardBackup) -> MaxValue
    """
    if isinstance(backup, PercentileRewardBackup):
        return ExpectedValue()
    return MaxValue()
