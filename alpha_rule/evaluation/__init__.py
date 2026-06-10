"""
Evaluation package.

The stable surface every evaluator implements: the ``Evaluator`` protocol
plus ``EvalResult`` and ``RuleStringNode``. Concrete evaluators
(``NeuralEvaluator``, ``RuleSimulator``) arrive in later components.
"""
from alpha_rule.evaluation.evaluator import (  # noqa: F401
    EvalResult,
    Evaluator,
    RuleStringNode,
)

__all__ = ["EvalResult", "Evaluator", "RuleStringNode"]
