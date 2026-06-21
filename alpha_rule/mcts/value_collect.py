"""
Value-sample collection seam.

After a search round, a ``ValueSampleCollector`` walks the (already-built) MCTS
tree and returns ``(state_name, raw_value_target)`` pairs to train the value head
on, beyond the single committed step. This amortizes one expensive search over
many value samples without re-searching.

``TreeQmaxCollector`` is the default: it emits ``node.Q_max`` (best rule reachable
from that node) for every explored, non-dead node with ``N >= min_visits``. The
target is optimistic by construction (a max), which is what makes a shallow
prefix that leads to a great rule look promising; ``min_visits`` drops nodes whose
``Q_max`` rests on a single lucky rollout. Other collectors (sim re-eval,
random-completion) can be dropped in behind the same protocol.
"""
from __future__ import annotations

import math
from typing import List, Optional, Protocol, Tuple, runtime_checkable


@runtime_checkable
class ValueSampleCollector(Protocol):
    """Maps a search root to ``(state_name, raw_value_target)`` pairs."""

    def collect(self, root) -> List[Tuple[str, float]]: ...


class TreeQmaxCollector:
    """Harvest ``(name, Q_max)`` for explored, non-dead descendants of ``root``.

    The root itself is skipped (it is already the committed step). Nodes with
    fewer than ``min_visits`` visits, or a non-finite ``Q_max`` (unexplored or
    dead), are excluded.
    """

    def __init__(self, min_visits: int = 2):
        if min_visits < 1:
            raise ValueError(f"min_visits must be >= 1, got {min_visits!r}")
        self.min_visits = min_visits

    def collect(self, root) -> List[Tuple[str, float]]:
        out: List[Tuple[str, float]] = []
        stack = list(getattr(root, "children", []))
        while stack:
            node = stack.pop()
            if getattr(node, "is_dead", False):
                continue            # dead subtree: skip it and its descendants
            q = node.Q_max
            if node.N >= self.min_visits and math.isfinite(q):
                out.append((node.name, float(q)))
            stack.extend(node.children)
        return out
