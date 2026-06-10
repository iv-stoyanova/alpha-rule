"""
Selection strategy seam.

A ``SelectionStrategy`` picks the next child to descend into during MCTS.
The AlphaZero loop uses ``PUCTSelection``, which scores each child with
the PUCT rule using its ``.prior`` (populated by ``NeuralEvaluator``).

The protocol is kept tiny on purpose: ``select(parent) -> Optional[node]``
plus an auxiliary ``score(parent, child) -> float`` for diagnostics and
the search-tree visualisation.
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
    AlphaZero-style PUCT with KataGo-style FPU.

    Score per child:

        Q(s, a) + c_puct * P(s, a) * sqrt(sum_N(s)) / (1 + N(s, a))

    where:

        Q(s, a) = ``child.Q_max`` for visited children. For *unvisited*
                  children we use a "first-play urgency" estimate:

                      FPU = parent_Q - fpu_reduction * sqrt(visited_policy_mass)

                  where ``parent_Q`` is the parent's own Q_max (or 0 if
                  the parent has no statistic yet) and
                  ``visited_policy_mass`` is the sum of priors over
                  already-visited siblings. KataGo (Wu 2019) introduced
                  this in place of AlphaGo-Zero's FPU = 0 because, in
                  positive-reward domains, FPU = 0 makes unvisited
                  children look strictly worse than any visited sibling
                  and PUCT degenerates into a depth-first sweep.
        P(s, a) = ``child.prior``, populated by ``NeuralEvaluator``
                  through the search loop.
        sum_N(s) = total visits across siblings (using max(1, …) so the
                   first call doesn't square-root zero).

    Plug-compatible with ``SelectionStrategy``; the self-play loop calls
    ``selection.select(parent)``, so a different strategy drops in without
    touching the search.
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
        """Return the node's Q-estimate under the configured ``q_source``,
        or ``None`` when no usable statistic exists yet.

        ``"max"`` reads ``node.Q_max`` (default; matches
        ``MaxRewardBackup``). ``"filtered_mean"`` reads
        ``node.Q_sum / node.N_passers`` (matches ``PercentileRewardBackup``, which uses the proper percentile-filtered mean instead of ``Q_max``,
        which would otherwise ignore the threshold entirely)."""
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
