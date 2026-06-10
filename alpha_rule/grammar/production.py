"""
Grammar productions.

A ``Production`` is the smallest unit a ``Grammar`` exposes: one legal move
from a state, named by the token it appends.

Fields:
    name: the token this production appends. It doubles as the vocabulary
        token the network tokenizer sees, so it must be unique across the
        productions a single grammar exposes.
    kind: a free-form label the grammar uses to tell its own productions
        apart inside ``apply`` (for example "event", "relation", "terminal"
        in the bundled Allen grammar). The MCTS and the network never read
        ``kind``: only the grammar that created the production does.

Because nothing here is specific to Allen intervals, ``Production`` is fully
generic. To plug in a different formal language you reuse it unchanged and
give your grammar's productions whatever ``kind`` labels help you build the
next state (one label, or several, or the conventional ones below). You do
not need to edit this file.

Conventional labels used by the bundled ``AllenIntervalGrammar``:
    "event"    appends an event-type token (for example "A", "B")
    "relation" appends an Allen-interval relation token (for example "<", "m")
    "terminal" finishes the formula (the special END_RULE token)

Example:
    >>> from alpha_rule.grammar.production import Production
    >>> Production(name="A", kind="event")
    Production(name='A', kind='event')
    >>> Production(name="END_RULE", kind="terminal")
    Production(name='END_RULE', kind='terminal')
    >>> # a different grammar is free to use its own labels:
    >>> Production(name="digit", kind="token")
    Production(name='digit', kind='token')
"""
from __future__ import annotations

from dataclasses import dataclass


# A ``kind`` is any non-empty string. This alias documents intent at call
# sites; it does not restrict the value, since the core never inspects it.
ProductionKind = str


@dataclass(frozen=True)
class Production:
    """One grammar production: the token to append plus a grammar-private kind."""

    name: str
    kind: ProductionKind

    def __post_init__(self):
        if not self.name:
            raise ValueError("Production.name cannot be empty")
        if not self.kind:
            raise ValueError("Production.kind cannot be empty")
