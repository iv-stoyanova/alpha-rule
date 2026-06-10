"""
Allen-interval grammar: the concrete ``Grammar`` for temporal rules.

This module is the only place that knows the Allen-interval construction
rules:

    * event steps add an event-type token (the triangular-number schedule),
    * every other step adds an Allen relation token,
    * every non-root state may also fire ``END_RULE`` to finish the rule.

It owns both the legal action set (``applicable_productions``) and the state
transition (``apply`` builds the successor node and its ``AllenMatrix``). The
MCTS node carries none of this, so swapping this class for another
``Grammar`` leaves the search and the network unchanged.
"""
from __future__ import annotations

import math
from typing import List, Sequence

from alpha_rule.grammar.grammar import Grammar
from alpha_rule.grammar.production import Production
from alpha_rule.helpers.matrix_operations import AllenRelation


END_RULE = "END_RULE"

DEFAULT_RELATIONS = tuple(AllenRelation.all_relations())


def should_add_event(level: int) -> bool:
    """
    Whether construction step ``level`` adds an event type (True) or an
    Allen relation (False).

    Event positions follow the triangular-number schedule: 0, 1, 3, 6, 10,
    15, ... are event indices; all others are relations.
    """
    if level == 0:
        return True
    step = level + 1
    n = (-1 + math.sqrt(1 + 8 * (step - 1))) / 2
    return n.is_integer()


class AllenIntervalGrammar(Grammar):
    """
    Concrete grammar for Allen-interval temporal rules.

    Args:
        event_types: iterable of event-type strings, e.g. ``("A", "B", "C")``.
        relations:   iterable of Allen relation symbols. Defaults to all 13.
    """

    def __init__(
        self,
        event_types: Sequence[str],
        relations: Sequence[str] = DEFAULT_RELATIONS,
    ):
        self.event_types: List[str] = list(event_types)
        self.relations: List[str] = list(relations)
        self._event_productions = [Production(name=t, kind="event") for t in self.event_types]
        self._relation_productions = [Production(name=r, kind="relation") for r in self.relations]
        self._end_rule = Production(name=END_RULE, kind="terminal")

    # ------------------------------------------------------------------ #
    # Grammar protocol
    # ------------------------------------------------------------------ #

    def root(self):
        """
        Fresh start state: an empty ``<ROOT>`` node.

        The root is always an event step and never offers ``END_RULE``, so its
        number of legal actions is exactly the number of event types. That is
        known up front, so we pass it straight to the constructor instead of
        building the node and counting afterwards.
        """
        from alpha_rule.mcts.node import MCTSRuleNode  # local import: avoid cycle

        return MCTSRuleNode(
            name="<ROOT>",
            level=0,
            n_possible_actions=len(self.event_types),
        )

    def vocab(self) -> List[str]:
        """Token list used by ``GrammarTokenizer``."""
        return list(self.event_types) + list(self.relations) + [END_RULE]

    def applicable_productions(self, state) -> List[Production]:
        """
        Legal next productions at ``state``. ``END_RULE`` first (for any
        non-root, non-terminal state), then events or relations depending on
        the construction-step schedule. A terminal state has none.
        """
        if getattr(state, "is_terminal", False):
            return []
        if should_add_event(state.level):
            base = list(self._event_productions)
        else:
            base = list(self._relation_productions)
        if state.name != "<ROOT>":
            return [self._end_rule] + base
        return base

    def apply(self, state, production: Production):
        """
        Apply ``production`` to ``state``: build the successor node, link it
        under ``state``, and return it. Owns the Allen name building and
        ``AllenMatrix`` construction.

        Raises ``ValueError`` if ``production`` does not match the step the
        schedule is due for (an event where a relation is expected, or the
        other way round).
        """
        from alpha_rule.rules.allen_matrix import AllenMatrix

        if production.kind == "terminal":
            # END_RULE keeps the parent's matrix and yields a terminal node.
            return self._child(
                state,
                production,
                name=f"{state.name} <END>",
                rule=state.rule,
                is_terminal=True,
            )

        expected = "event" if should_add_event(state.level) else "relation"
        if production.kind != expected:
            raise ValueError(
                f"level {state.level} expects a production of kind {expected!r}, "
                f"got {production.kind!r} ({production.name!r})"
            )

        # Keep the first ``level`` tokens of the parent name (dropping any
        # unfilled placeholder slots), then append the new token.
        name_prefix = "" if state.level == 0 else " ".join(state.name.split()[: state.level])
        new_rule_str = f"{name_prefix} {production.name}".strip()

        new_matrix = AllenMatrix.from_hierarchy_string(new_rule_str)
        return self._child(
            state,
            production,
            name=new_matrix.get_hierarchy_string(),
            rule=new_matrix,
            is_terminal=False,
        )

    def is_terminal(self, state) -> bool:
        return bool(getattr(state, "is_terminal", False))

    # ------------------------------------------------------------------ #
    # Node construction
    # ------------------------------------------------------------------ #

    def _child(self, parent, production: Production, *, name, rule, is_terminal):
        """
        Build a child node, link it under ``parent``, stamp the number of
        legal actions the grammar allows at the child, and return it.

        This is the single place node creation, the parent link, and the
        ``n_possible_actions`` count live, so ``root`` and ``apply`` stay
        short.
        """
        from alpha_rule.mcts.node import MCTSRuleNode  # local import: avoid cycle

        child = MCTSRuleNode(
            name=name,
            level=parent.level + 1,
            parent=parent,
            parent_action=production.name,
            rule=rule,
            is_terminal=is_terminal,
        )
        child.n_possible_actions = len(self.applicable_productions(child))
        parent.children.append(child)
        return child
