"""
Neural network package for AlphaZero-style search.

Lazy by design: importing this module does not import torch. Each
sub-module imports torch at its own top, so call-sites that only need
the tokeniser (which never instantiates a tensor without ``encode``)
can use it standalone.
"""
from alpha_rule.nn.tokenizer import GrammarTokenizer  # noqa: F401

__all__ = ["GrammarTokenizer"]


def __getattr__(name: str):
    """Lazy resolution for the torch-dependent symbols."""
    if name == "FormulaEncoder":
        from alpha_rule.nn.encoder import FormulaEncoder
        return FormulaEncoder
    if name in {"PolicyHead", "ValueHead"}:
        from alpha_rule.nn.heads import PolicyHead, ValueHead
        return {"PolicyHead": PolicyHead, "ValueHead": ValueHead}[name]
    if name == "AllenFormulaNet":
        from alpha_rule.nn.model import AllenFormulaNet
        return AllenFormulaNet
    if name in {"train_step", "collate", "TrainStepLog"}:
        from alpha_rule.nn import training
        return getattr(training, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
