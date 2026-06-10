"""
Expansion strategy seam.

An ``ExpansionStrategy`` adds one child to a parent and returns it (or
``None`` when the parent is fully expanded).

``RuleExpansion`` is grammar-driven: it expands the parent's productions in
the grammar's order and lets ``grammar.apply`` build and attach each child.
All grammar-specific knowledge stays in the grammar, so swapping the grammar
swaps the expansion behaviour too.
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
        parent already has a child for every applicable production.

        Children are appended in the grammar's production order and never
        removed, so the next production to expand is simply the one at index
        ``len(parent.children)`` -- no need to rebuild a set of already-tried
        actions on every call."""
        productions = self.grammar.applicable_productions(parent)
        idx = len(parent.children)
        if idx < len(productions):
            return self.grammar.apply(parent, productions[idx])
        return None
