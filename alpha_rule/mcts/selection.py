"""
Selection strategy seam.

A ``SelectionStrategy`` picks the next child to descend into during a
simulation. ``PUCTSelection`` is the default. The protocol is two methods:
``select(parent)`` returns the chosen child (or None), and
``score(parent, child)`` exposes the per-child score for diagnostics.
"""
from __future__ import annotations

import math
from typing import Optional, Protocol, runtime_checkable


@runtime_checkable
class SelectionStrategy(Protocol):
    """Picks the next child during MCTS selection."""

    def score(self, parent, child) -> float: ...

    def select(self, parent) -> Optional[object]: ...


class PUCTSelection(SelectionStrategy):
    """
    PUCT selection with first-play urgency for unvisited children.

    Each child scores ``Q + c_puct * prior * sqrt(sum_N) / (1 + N)``:

        Q       ``child.Q_max`` (or the filtered mean under
                ``q_source="filtered_mean"``) for a visited child. For an
                unvisited child, a first-play-urgency (FPU) estimate
                ``parent_Q - fpu_reduction * sqrt(visited_prior_mass)``, where
                ``visited_prior_mass`` is the prior mass already explored under
                the parent. (Plain FPU = 0 makes an unvisited child look worse
                than any visited one when rewards are positive, turning the
                search depth-first; this reduction is the KataGo fix.)
        prior   ``child.prior``, set by the network (1.0 until then).
        sum_N   total visits across the siblings, floored at 1.

    ``q_source`` ("max" or "filtered_mean") selects which Q the score reads,
    so it can match the backup in use.
    """

    def __init__(
        self,
        c_puct: float = 1.5,
        fpu_reduction: float = 0.25,
        q_source: str = "max",
    ):
        if fpu_reduction < 0.0:
            raise ValueError(
                f"fpu_reduction must be >= 0, got {fpu_reduction!r}"
            )
        if q_source not in ("max", "filtered_mean"):
            raise ValueError(
                f"unknown q_source: {q_source!r}; "
                "expected 'max' or 'filtered_mean'"
            )
        self.c_puct = c_puct
        self.fpu_reduction = fpu_reduction
        self.q_source = q_source

    def _extract_q(self, node) -> Optional[float]:
        """Return the node's Q under the configured ``q_source``, or ``None``
        when it has no value yet.

        ``"max"`` reads ``node.Q_max`` (matches ``MaxRewardBackup``);
        ``"filtered_mean"`` reads ``node.Q_sum / node.N_passers`` (matches
        ``PercentileRewardBackup``). ``None`` means unvisited, or that no
        sample has passed the percentile threshold."""
        if self.q_source == "max":
            return node.Q_max if node.Q_max != float("-inf") else None
        # q_source == "filtered_mean"
        return (node.Q_sum / node.N_passers) if node.N_passers > 0 else None

    def score(self, parent, child) -> float:
        if child.is_dead:
            return float("-inf")
        sum_n = max(1, sum(c.N for c in parent.children))
        q = self._extract_q(child) if child.N > 0 else None
        if q is None:
            # KataGo-style FPU: estimate the unvisited child's value at
            # the parent's known Q, reduced by how much prior mass has
            # already been explored under the parent. Reads the same
            # ``q_source`` so the FPU's scale matches the visited-child
            # Q scale; falls back to 0 when the parent has no statistic
            # yet (very first simulation at the search root).
            parent_q = self._extract_q(parent)
            if parent_q is None:
                parent_q = 0.0
            visited_prior_mass = sum(
                c.prior for c in parent.children if c.N > 0
            )
            q = parent_q - self.fpu_reduction * math.sqrt(visited_prior_mass)
        u = self.c_puct * child.prior * math.sqrt(sum_n) / (1 + child.N)
        return q + u

    def select(self, parent):
        non_dead = [c for c in parent.children if not c.is_dead]
        if not non_dead:
            return None
        return max(non_dead, key=lambda c: self.score(parent, c))
