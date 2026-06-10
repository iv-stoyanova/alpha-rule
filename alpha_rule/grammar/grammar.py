"""
``Grammar`` protocol: the single source of truth for the search space.

A grammar fully defines what the MCTS searches over. To plug in a different
formal language, implement these five methods and pass an instance to
``run_self_play`` / ``train``. Nothing in ``mcts`` or ``nn`` knows about
Allen intervals, so the search and the network stay unchanged.

    root()                        the start state (a fresh node)
    vocab()                       every token the network tokenizer needs
    applicable_productions(state) which productions are legal at this state
    apply(state, production)      deterministic transition from s to s'
                                  (builds the child node, links it under
                                  ``state``, and stamps the child's
                                  ``n_possible_actions``)
    is_terminal(state)            the stopping check

The protocol is ``runtime_checkable`` so call sites can use ``isinstance``
without nominal subclassing. ``state`` is typed loosely on purpose: the
search threads ``MCTSRuleNode`` instances through, but the grammar only
relies on the small surface above and never reaches into MCTS statistics.

Example: a minimal grammar over binary strings, with no Allen logic. Any
object exposing these five methods satisfies the protocol.

    from alpha_rule.grammar.production import Production
    from alpha_rule.mcts.node import MCTSRuleNode

    class BinaryStrings:
        TOKENS = ("0", "1")
        MAX_LEN = 3

        def root(self):
            return MCTSRuleNode(name="<ROOT>", level=0, n_possible_actions=2)

        def vocab(self):
            return list(self.TOKENS) + ["END"]

        def applicable_productions(self, state):
            if state.is_terminal or state.level >= self.MAX_LEN:
                return []
            moves = [Production(name=t, kind="token") for t in self.TOKENS]
            if state.name != "<ROOT>":          # END only after the first token
                moves = [Production(name="END", kind="stop")] + moves
            return moves

        def apply(self, state, production):
            stop = production.kind == "stop"
            base = "" if state.name == "<ROOT>" else state.name
            name = (base + " <END>").strip() if stop else (base + " " + production.name).strip()
            child = MCTSRuleNode(
                name=name, level=state.level + 1, parent=state,
                parent_action=production.name, is_terminal=stop,
            )
            child.n_possible_actions = len(self.applicable_productions(child))
            state.children.append(child)
            return child

        def is_terminal(self, state):
            return state.is_terminal

The ``kind`` labels here ("token", "stop") are this grammar's own choice; the
core never reads them, only this ``apply`` does.
"""
from __future__ import annotations

from typing import List, Protocol, runtime_checkable

from alpha_rule.grammar.production import Production


@runtime_checkable
class Grammar(Protocol):
    def root(self): ...
    def vocab(self) -> List[str]: ...
    def applicable_productions(self, state) -> List[Production]: ...
    def apply(self, state, production: Production): ...
    def is_terminal(self, state) -> bool: ...
