"""
Grammar package — the single source of truth for the search space.

The grammar defines the productions, the start state, the transitions,
and the terminal check. The MCTS node carries no grammar logic, so
swapping in a different ``Grammar`` (see ``grammar.grammar.Grammar``)
changes what the search explores without touching ``mcts`` / ``nn``.
``AllenIntervalGrammar`` is the concrete Allen-interval implementation.
"""
from alpha_rule.grammar.allen import AllenIntervalGrammar  # noqa: F401
from alpha_rule.grammar.grammar import Grammar  # noqa: F401
from alpha_rule.grammar.production import Production, ProductionKind  # noqa: F401

__all__ = [
    "AllenIntervalGrammar",
    "Grammar",
    "Production",
    "ProductionKind",
]
