"""
Tests for the grammar package — now the SINGLE source of truth for the
search space.

Pins:
    - ``root()`` yields a ``<ROOT>`` start state offering only event
      productions (no ``END_RULE``).
    - ``applicable_productions`` prepends ``END_RULE`` for non-root,
      non-terminal states and uses the event/relation schedule.
    - ``vocab()`` covers every token plus ``END_RULE``.
    - ``apply`` builds the successor node (name + matrix), links it, and
      stamps ``n_possible_actions``; ``END_RULE`` yields a terminal.
    - A *different* grammar can drive ``run_self_play`` unchanged — the
      MCTS/NN core carries no Allen-specific assumptions. (Swap test.)
"""
from __future__ import annotations

import numpy as np

from alpha_rule.grammar.allen import AllenIntervalGrammar, should_add_event
from alpha_rule.grammar.production import Production


# --------------------------------------------------------------------------- #
# Vocab
# --------------------------------------------------------------------------- #

def test_vocab_includes_event_types_relations_and_end_rule():
    g = AllenIntervalGrammar(event_types=("A", "B"), relations=("<", ">"))
    vocab = g.vocab()
    assert {"A", "B", "<", ">", "END_RULE"} <= set(vocab)


# --------------------------------------------------------------------------- #
# root() + applicable_productions
# --------------------------------------------------------------------------- #

def test_root_is_root_node_offering_only_events():
    g = AllenIntervalGrammar(event_types=("A", "B"), relations=("<", ">"))
    root = g.root()
    assert root.name == "<ROOT>"
    assert root.level == 0
    prods = g.applicable_productions(root)
    assert [p.name for p in prods] == ["A", "B"]          # no END_RULE on root
    assert all(p.kind == "event" for p in prods)
    assert root.n_possible_actions == 2


def test_non_root_prepends_end_rule():
    g = AllenIntervalGrammar(event_types=("A", "B"), relations=("<", ">"))
    root = g.root()
    child = g.apply(root, g.applicable_productions(root)[0])   # "A", level 1
    names = [p.name for p in g.applicable_productions(child)]
    assert names[0] == "END_RULE"


def test_production_kind_matches_event_or_relation_schedule():
    g = AllenIntervalGrammar(event_types=("A", "B"), relations=("<", ">"))
    root = g.root()
    a = g.apply(root, g.applicable_productions(root)[0])       # level 1
    end = next(p for p in g.applicable_productions(a) if p.name == "END_RULE")
    assert end.kind == "terminal"
    others = [p for p in g.applicable_productions(a) if p.name != "END_RULE"]
    expected_kind = "event" if should_add_event(a.level) else "relation"
    assert all(p.kind == expected_kind for p in others)


# --------------------------------------------------------------------------- #
# apply()
# --------------------------------------------------------------------------- #

def test_apply_event_sets_name_and_links_child():
    g = AllenIntervalGrammar(event_types=("A", "B"), relations=("<",))
    root = g.root()
    prod = next(p for p in g.applicable_productions(root) if p.name == "A")
    child = g.apply(root, prod)
    assert child.name == "A"
    assert child.parent is root
    assert child.parent_action == "A"
    assert child in root.children
    assert child.n_possible_actions == len(g.applicable_productions(child))


def test_apply_end_rule_marks_child_terminal_with_no_productions():
    g = AllenIntervalGrammar(event_types=("A",), relations=("<",))
    root = g.root()
    a = g.apply(root, g.applicable_productions(root)[0])
    end_prod = next(p for p in g.applicable_productions(a) if p.name == "END_RULE")
    terminal = g.apply(a, end_prod)
    assert terminal.is_terminal is True
    assert terminal.name.endswith("<END>")
    assert g.is_terminal(terminal) is True
    assert g.is_terminal(root) is False
    assert g.applicable_productions(terminal) == []           # nothing follows END


def test_n_possible_actions_matches_applicable_productions_for_each_node_kind():
    """
    The grammar stamps ``n_possible_actions`` on every node it builds. It is
    computed arithmetically in ``AllenIntervalGrammar._child`` for speed, so
    pin it against the real ``applicable_productions`` length for each kind of
    node: the root (event step, no END_RULE), an event-step child and a
    relation-step child (both offer END_RULE), and a terminal (offers nothing).
    If the arithmetic ever drifts from ``applicable_productions``,
    ``is_fully_expanded`` and the dead-cascade break silently.
    """
    g = AllenIntervalGrammar(event_types=("A", "B"), relations=("<", ">"))

    root = g.root()                                  # level 0: event step, no END_RULE
    after_one_event = g.apply(                       # level 1: event step, END_RULE offered
        root, next(p for p in g.applicable_productions(root) if p.name == "A")
    )
    after_two_events = g.apply(                      # level 2: relation step, END_RULE offered
        after_one_event,
        next(p for p in g.applicable_productions(after_one_event) if p.name == "B"),
    )
    terminal = g.apply(                              # terminal: nothing follows
        after_one_event,
        next(p for p in g.applicable_productions(after_one_event) if p.name == "END_RULE"),
    )

    for node in (root, after_one_event, after_two_events, terminal):
        assert node.n_possible_actions == len(g.applicable_productions(node)), node.name

    # Guard that the three kinds really were exercised.
    assert should_add_event(root.level) and should_add_event(after_one_event.level)
    assert not should_add_event(after_two_events.level)
    assert terminal.is_terminal and g.applicable_productions(terminal) == []


# --------------------------------------------------------------------------- #
# Swap test: a different Grammar drives run_self_play unchanged.
# --------------------------------------------------------------------------- #

class _ToyGrammar:
    """
    Minimal swap-in grammar with no Allen semantics: builds binary strings
    up to length 3, with an ``END`` terminal available after the first
    token. Implements the full ``Grammar`` protocol over plain
    ``MCTSRuleNode`` state objects — proving the MCTS/NN core makes no
    Allen-specific assumptions.
    """

    TOKENS = ("0", "1")
    MAX_LEN = 3

    def _new(self, *, name, level, parent=None, parent_action=None, is_terminal=False):
        from alpha_rule.mcts.node import MCTSRuleNode
        node = MCTSRuleNode(
            name=name, level=level, parent=parent,
            parent_action=parent_action, is_terminal=is_terminal,
        )
        node.n_possible_actions = len(self.applicable_productions(node))
        return node

    def root(self):
        return self._new(name="<ROOT>", level=0)

    def vocab(self):
        return list(self.TOKENS) + ["END"]

    def applicable_productions(self, state):
        if getattr(state, "is_terminal", False) or state.level >= self.MAX_LEN:
            return []
        prods = [Production(name=t, kind="event") for t in self.TOKENS]
        if state.name != "<ROOT>":
            prods = [Production(name="END", kind="terminal")] + prods
        return prods

    def apply(self, state, production):
        is_term = production.kind == "terminal"
        base = "" if state.name == "<ROOT>" else state.name
        name = (base + " <END>").strip() if is_term else (base + " " + production.name).strip()
        child = self._new(
            name=name, level=state.level + 1, parent=state,
            parent_action=production.name, is_terminal=is_term,
        )
        state.children.append(child)
        return child

    def is_terminal(self, state):
        return bool(getattr(state, "is_terminal", False))


class _LenSimulator:
    """Reward = token count of the rule name (favours longer strings)."""
    def evaluate(self, node):
        from alpha_rule.evaluation.evaluator import EvalResult
        toks = [t for t in node.name.replace("<END>", "").split() if t]
        return EvalResult(value=float(len(toks)))


def test_custom_grammar_drives_self_play_unchanged():
    import pytest
    run_self_play = pytest.importorskip("alpha_rule.mcts.self_play").run_self_play

    traj = run_self_play(
        grammar=_ToyGrammar(),
        simulator=_LenSimulator(),
        n_simulations=8,
        depth_limit=3,
        rng=np.random.default_rng(0),
    )
    assert len(traj.steps) >= 1
    # Every produced token comes from the toy grammar's vocab — no Allen
    # symbols leaked in from the MCTS/NN core.
    allowed = set(_ToyGrammar().vocab()) | {"<END>"}
    for step in traj.steps:
        toks = step.next_state.split() if step.next_state else []
        assert all(t in allowed for t in toks), f"unexpected token in {step.next_state!r}"
