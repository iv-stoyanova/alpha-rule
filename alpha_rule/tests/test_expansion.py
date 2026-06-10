"""
Tests for ``mcts.expansion.RuleExpansion`` — the grammar-driven child
creation seam that replaced ``MCTSRuleNode.expand``.

Pins:
    - ``expand`` builds children in the grammar's production order and
      returns ``None`` once the parent is fully expanded.
    - ``is_fully_expanded`` tracks ``n_possible_actions`` (stamped by the
      grammar), so it agrees with "expand returned None".
    - For a NON-root node the first child is the ``END_RULE`` terminal,
      then the events/relations.
    - No production is expanded twice.
"""
from __future__ import annotations

from alpha_rule.grammar.allen import AllenIntervalGrammar
from alpha_rule.mcts.expansion import RuleExpansion


def _grammar():
    return AllenIntervalGrammar(event_types=("A", "B"), relations=("<",))


def test_root_expands_into_events_in_order_then_none():
    g = _grammar()
    exp = RuleExpansion(g)
    root = g.root()
    assert not root.is_fully_expanded()

    first = exp.expand(root)
    second = exp.expand(root)
    assert first.parent_action == "A" and first.name == "A"
    assert second.parent_action == "B" and second.name == "B"
    assert root.is_fully_expanded()
    assert exp.expand(root) is None              # exhausted


def test_non_root_first_child_is_end_rule_terminal():
    g = _grammar()
    exp = RuleExpansion(g)
    root = g.root()
    a = exp.expand(root)                          # level-1 node "A"

    terminal = exp.expand(a)                      # END_RULE is first applicable
    assert terminal.parent_action == "END_RULE"
    assert terminal.is_terminal is True
    assert terminal.name.endswith("<END>")

    nxt = exp.expand(a)                           # then the first event/relation
    assert nxt.parent_action != "END_RULE"
    assert nxt.is_terminal is False


def test_is_fully_expanded_matches_n_possible_actions():
    g = _grammar()
    exp = RuleExpansion(g)
    a = exp.expand(g.root())                      # level-1 node
    assert a.n_possible_actions == len(g.applicable_productions(a))
    count = 0
    while not a.is_fully_expanded():
        assert exp.expand(a) is not None
        count += 1
    assert count == a.n_possible_actions
    assert exp.expand(a) is None


def test_no_production_expanded_twice():
    g = _grammar()
    exp = RuleExpansion(g)
    root = g.root()
    while not root.is_fully_expanded():
        exp.expand(root)
    actions = [c.parent_action for c in root.children]
    assert len(actions) == len(set(actions))     # each production exactly once
