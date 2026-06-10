"""
Evaluation package.

The stable surface every evaluator implements: the ``Evaluator`` protocol plus
``EvalResult`` and ``RuleStringNode``. ``NeuralEvaluator`` wraps an
``AllenFormulaNet`` (resolved lazily so importing this package stays light and
torch-free); the reinforcement-learning ``RuleSimulator`` arrives with the
``[rl]`` backend.
"""
from alpha_rule.evaluation.evaluator import (  # noqa: F401
    EvalResult,
    Evaluator,
    RuleStringNode,
)

__all__ = [
    "EvalResult",
    "Evaluator",
    "RuleStringNode",
    "NeuralEvaluator",
    "DEFAULT_VALUE_SCALE",
]


def __getattr__(name: str):
    """Lazily resolve the concrete neural evaluator (keeps the eager import of
    this package from pulling in ``grammar`` / ``mcts`` and torch)."""
    if name in {"NeuralEvaluator", "DEFAULT_VALUE_SCALE"}:
        from alpha_rule.evaluation import neural_evaluator
        return getattr(neural_evaluator, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
