"""
Expansion strategy seam.

An ``ExpansionStrategy`` adds one new child to a parent node and returns it
(or ``None`` when the parent is fully expanded).

``RuleExpansion`` is grammar-driven: it asks the grammar which productions
are legal at the parent, picks the first one not yet expanded (in the
grammar's own order), and lets ``grammar.apply`` build + attach the child.
All Allen-specific knowledge therefore lives in the grammar, not here and
not on the node, so swapping the grammar swaps the expansion behaviour too.
"""
from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

from alpha_rule.grammar.grammar import Grammar


@runtime_checkable
class ExpansionStrategy(Protocol):
    """Adds one child to ``parent`` and returns it, or None if exhausted."""

    def expand(self, parent) -> Optional[object]: ...


class RuleExpansion(ExpansionStrategy):
    """Expand a node via its grammar (the single source of productions)."""

    def __init__(self, grammar: Grammar):
        self.grammar = grammar

    def expand(self, parent):
        """Build the next un-tried production's child, or ``None`` if the
        parent already has a child for every applicable production."""
        expanded = {child.parent_action for child in parent.children}
        for production in self.grammar.applicable_productions(parent):
            if production.name not in expanded:
                return self.grammar.apply(parent, production)
        return None
