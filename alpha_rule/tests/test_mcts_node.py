"""
Tests for ``MCTSRuleNode`` — now a PURE STATE container.

The node holds statistics, tree links, and flags; it has no Allen logic,
no selection/expansion/backprop behaviour (those live in ``grammar`` /
``mcts.selection`` / ``mcts.expansion`` / ``mcts.backprop``).

Pins:
    - Construction defaults (fresh stats, ``prior=1.0``, alive, not terminal).
    - ``is_fully_expanded`` is driven solely by ``n_possible_actions`` vs
      the number of children (so the backprop dead-cascade can ask "are
      all children present?" without a grammar reference).
    - The removed behaviour methods are really gone (guards against
      accidental reintroduction).
"""
from __future__ import annotations

import numpy as np

from alpha_rule.mcts.node import MCTSRuleNode


def test_construction_defaults():
    n = MCTSRuleNode(name="<ROOT>")
    assert n.name == "<ROOT>"
    assert n.level == 0
    assert n.parent is None
    assert n.parent_action is None
    assert n.children == []
    assert n.is_terminal is False
    assert n.is_dead is False
    # Fresh MCTS statistics.
    assert n.N == 0
    assert n.Q == 0.0
    assert n.Q_max == -np.inf
    assert n.Q_sum == 0.0
    assert n.N_passers == 0
    assert n.past_rewards == []
    # AlphaZero prior defaults to 1.0 (uniform-ish until the net writes it).
    assert n.prior == 1.0


def test_is_fully_expanded_tracks_n_possible_actions():
    parent = MCTSRuleNode(name="<ROOT>", n_possible_actions=2)
    assert parent.is_fully_expanded() is False
    parent.children.append(MCTSRuleNode(name="A", parent=parent, parent_action="A"))
    assert parent.is_fully_expanded() is False
    parent.children.append(MCTSRuleNode(name="B", parent=parent, parent_action="B"))
    assert parent.is_fully_expanded() is True


def test_zero_action_node_is_immediately_fully_expanded():
    # A terminal-style node the grammar stamped with 0 possible actions.
    leaf = MCTSRuleNode(name="A <END>", is_terminal=True, n_possible_actions=0)
    assert leaf.is_fully_expanded() is True


def test_node_is_pure_state_no_behaviour_methods():
    n = MCTSRuleNode(name="<ROOT>")
    for removed in (
        "expand", "untried_actions", "ucb_score", "select", "backpropagate",
        "node_value", "set_value_mode", "mean_reward",
    ):
        assert not hasattr(n, removed), f"node should not carry {removed!r}"
    # No Allen-specific construction attributes leaked back in.
    for attr in ("possible_event_types", "allen_relations", "value_mode", "c"):
        assert not hasattr(n, attr), f"node should not carry {attr!r}"


def test_repr_renders_node_and_children():
    root = MCTSRuleNode(name="<ROOT>", n_possible_actions=1)
    child = MCTSRuleNode(name="A", parent=root, parent_action="A")
    child.N = 3
    child.Q_max = 1.5
    root.children.append(child)
    text = repr(root)
    assert "<ROOT>" in text and "A" in text and "N=3" in text
