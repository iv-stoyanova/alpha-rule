"""
MCTSRuleNode: one position in the search tree.

A node is pure state -- tree links, MCTS statistics, and a couple of flags. It
holds no grammar or search logic: the grammar builds and links nodes, and the
selection/expansion/backprop strategies update the statistics. Swapping the
grammar or a strategy therefore never touches this class.

The class docstring lists each field and where it is used.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np


class MCTSRuleNode:
    """
    A node in the rule-search MCTS tree. Pure state: it stores data and has
    no search behaviour of its own.

    Identity and structure:
        name:          the rule string built so far, for example "A B <". The
                       grammar writes it; the network tokenizes it.
        level:         construction depth. 0 is the empty ``<ROOT>``; each
                       applied production adds one.
        parent:        the node this one was expanded from (``None`` at root).
        parent_action: the production name on the edge from ``parent`` to here.
        children:      the child nodes already expanded under this one.
        rule:          opaque grammar payload for this state. The Allen grammar
                       stores the node's ``AllenMatrix`` here; another grammar
                       can store anything or nothing. The search never looks
                       inside it.
        is_terminal:   ``True`` once the rule is finished (``END_RULE``
                       applied); nothing follows a terminal node.
        is_dead:       ``True`` once the whole subtree is exhausted. The
                       backprop dead-cascade sets it so selection skips it.
        n_possible_actions:
                       how many productions the grammar allows at this node.
                       The grammar stamps it when it builds the node. It is the
                       only grammar-derived number stored here, which lets
                       ``is_fully_expanded`` check "are all children present?"
                       without a grammar reference.

    MCTS statistics:
        N:             visit count.
        Q:             running aggregate maintained by the backup (a running
                       max under ``MaxRewardBackup``, a running sum under
                       ``PercentileRewardBackup``). NOT read by selection or by
                       the value target -- those use ``Q_max`` /
                       ``Q_sum``/``N_passers``. Kept as a diagnostic.
        Q_max:         best value seen in this subtree; read by
                       ``PUCTSelection(q_source="max")`` and ``MaxValue``.
        Q_sum, N_passers:
                       sum and count of the values that counted under the
                       active backup strategy. Their ratio is the filtered mean
                       read by ``PUCTSelection(q_source="filtered_mean")`` and
                       the ``ExpectedValue`` / ``MeanPercentileValue`` targets.
        past_rewards:  per-node history of values, which
                       ``PercentileRewardBackup`` thresholds on.
        realized_reward:
                       the simulator's evaluation of this state's rule, stamped
                       by ``run_self_play`` when the node is the chosen step.
                       Read by the ``RealizedReturn`` value target. ``None``
                       until evaluated (e.g. the episode root never is).

    AlphaZero prior:
        prior:         P(s, a) for this child under its parent, written by
                       ``NeuralEvaluator`` and read by ``PUCTSelection``.
                       Defaults to 1.0 until a network sets it.
    """

    # Pure-state node held at every tree position, so drop the per-instance
    # __dict__: less memory per node and faster attribute reads/writes in
    # selection/backprop. Every attribute the node ever carries is listed here;
    # adding one elsewhere (e.g. a future search module) must be added here too.
    __slots__ = (
        "name", "level", "parent", "parent_action", "children", "rule",
        "is_terminal", "is_dead", "n_possible_actions",
        "N", "Q", "Q_max", "Q_sum", "N_passers", "past_rewards", "prior",
        "realized_reward",
    )

    def __init__(
        self,
        *,
        name: str,
        level: int = 0,
        parent: Optional["MCTSRuleNode"] = None,
        parent_action: Optional[str] = None,
        rule: object = None,
        is_terminal: bool = False,
        n_possible_actions: int = 0,
    ):
        # --- Identity / structure ------------------------------------- #
        self.name = name
        self.level = level
        self.parent = parent
        self.parent_action = parent_action
        self.children: List["MCTSRuleNode"] = []
        self.rule = rule                     # opaque payload (e.g. AllenMatrix)
        self.is_terminal = is_terminal
        self.is_dead = False

        # Productions the grammar allows here (stamped by the grammar).
        self.n_possible_actions = n_possible_actions

        # --- MCTS statistics ------------------------------------------ #
        self.N = 0
        self.Q = 0.0
        self.Q_max = -np.inf

        # Filtered-mean tracking: the proper mean of values that "counted"
        # under the active backup strategy (``Q_sum / N_passers``). Read by
        # ``PUCTSelection(q_source="filtered_mean")``. ``past_rewards`` is
        # the per-node history ``PercentileRewardBackup`` thresholds on.
        self.Q_sum: float = 0.0
        self.N_passers: int = 0
        self.past_rewards: List[float] = []

        # AlphaZero prior P(s, a) for this child under its parent.
        self.prior: float = 1.0

        # Ground-truth simulator reward of this state's rule, stamped by
        # ``run_self_play`` when this node is the chosen step. Read by the
        # ``RealizedReturn`` value target. ``None`` until evaluated.
        self.realized_reward: Optional[float] = None

    def is_fully_expanded(self) -> bool:
        """True once every production the grammar allows here has a child."""
        return len(self.children) >= self.n_possible_actions

    def __repr__(self, depth: int = 0) -> str:
        # ``depth`` is the indentation depth for the recursive tree print, not
        # the node's construction ``self.level`` (printed as "L{level}" below).
        indent = " " * (depth * 4)
        dead = ", DEAD" if self.is_dead else ""
        out = (
            f"{indent}{self.name} (L{self.level}) "
            f"[N={self.N}, Q_max={self.Q_max:.2f}, prior={self.prior:.3f}{dead}]\n"
        )
        for child in self.children:
            out += child.__repr__(depth + 1)
        return out
