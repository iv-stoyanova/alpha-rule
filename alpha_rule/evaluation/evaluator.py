"""
Evaluation protocol.

One surface every MCTS evaluator satisfies, so the search loop can score a
node the same way no matter what does the scoring: an RL agent, an
AlphaZero-style network, or a hand-written heuristic.

``EvalResult.value`` is the scalar quality signal backpropagated into the
tree. ``priors`` is an optional per-action probability that AlphaZero PUCT
selection uses for the newly expanded children; an evaluator that does not
produce priors leaves it ``None``.

The protocol is declared with ``runtime_checkable`` so callers can do
``isinstance(obj, Evaluator)`` without nominal inheritance.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Protocol, runtime_checkable


@dataclass
class EvalResult:
    """
    Outcome of evaluating a single MCTS node.

    Attributes:
        value: Scalar score to backpropagate up the tree. Higher is better.
            ``-np.inf`` means structural failure (the rule never matched).
        priors: Optional dict mapping each action name to a prior
            probability. Populated by AlphaZero-style evaluators; an evaluator
            that does not produce priors leaves it ``None``.
    """

    value: float
    priors: Optional[Dict[str, float]] = field(default=None)


@runtime_checkable
class Evaluator(Protocol):
    """Anything that can score an MCTS node."""

    def evaluate(self, node) -> EvalResult | float:
        """
        Score a single node.

        Returning a raw float is also accepted: the search loop wraps it in
        an ``EvalResult(value=...)`` internally.
        """
        ...


@dataclass(frozen=True)
class RuleStringNode:
    """
    Minimal node-like object exposing only ``.name``.

    Every ``Evaluator`` in this package reads ``node.name`` and nothing else.
    When a caller has just a rule string and wants to score it, wrapping the
    string in a ``RuleStringNode`` avoids constructing a full ``MCTSRuleNode``
    (which needs a grammar, event types, relations, a level, and so on).

    Example:
        >>> class LengthEvaluator:
        ...     def evaluate(self, node):
        ...         return EvalResult(value=float(len(node.name.split())))
        >>> LengthEvaluator().evaluate(RuleStringNode(name="A B <")).value
        3.0
    """

    name: str
