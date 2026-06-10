"""
MCTS package.

Component C2 brings ``MCTSRuleNode``, the pure-state search node. Later
components add the selection, expansion, backprop, replay, and self-play
modules and re-export them here.
"""
from alpha_rule.mcts.node import MCTSRuleNode  # noqa: F401

__all__ = ["MCTSRuleNode"]
