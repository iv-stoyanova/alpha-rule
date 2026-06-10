"""
Tests for ``evaluation.neural_evaluator.NeuralEvaluator``.

Pins:
    - ``evaluate(node)`` returns an ``EvalResult`` with finite ``value``
      and a ``priors`` dict whose keys are exactly the names of the
      currently-applicable productions.
    - Priors over applicable productions sum to 1 (within float tol).
    - Non-applicable productions never appear in the priors dict: the
      mask in the evaluator zeros them out.
    - Repeat calls on the same node give the same result (no hidden
      state; ``predict`` runs in eval mode).
"""
from __future__ import annotations

import torch

from alpha_rule.evaluation.neural_evaluator import NeuralEvaluator
from alpha_rule.grammar.allen import AllenIntervalGrammar
from alpha_rule.nn.model import AllenFormulaNet
from alpha_rule.nn.tokenizer import GrammarTokenizer

_GRAMMAR = AllenIntervalGrammar(event_types=("A", "B"), relations=("<",))


def _setup():
    torch.manual_seed(11)
    g = _GRAMMAR
    tok = GrammarTokenizer(g)
    model = AllenFormulaNet(tok, d_model=16, nhead=2, num_layers=1, max_len=12)
    evaluator = NeuralEvaluator(model, g, max_len=12)
    return evaluator, g


def _root():
    return _GRAMMAR.root()


def test_evaluate_returns_finite_value_and_priors_dict():
    evaluator, _ = _setup()
    result = evaluator.evaluate(_root())
    assert isinstance(result.value, float)
    assert result.value == result.value             # not NaN
    assert isinstance(result.priors, dict)
    assert len(result.priors) > 0


def test_priors_keys_match_applicable_productions():
    evaluator, g = _setup()
    root = _root()
    expected = {p.name for p in g.applicable_productions(root)}
    result = evaluator.evaluate(root)
    assert set(result.priors.keys()) == expected


def test_priors_sum_to_one():
    evaluator, _ = _setup()
    result = evaluator.evaluate(_root())
    assert abs(sum(result.priors.values()) - 1.0) < 1e-5


def test_priors_dict_excludes_non_applicable_productions():
    """At root, END_RULE is NOT applicable (only events are)."""
    evaluator, _ = _setup()
    result = evaluator.evaluate(_root())
    assert "END_RULE" not in result.priors           # excluded by grammar at root


def test_evaluate_is_deterministic_for_same_state():
    evaluator, _ = _setup()
    a = evaluator.evaluate(_root())
    b = evaluator.evaluate(_root())
    assert abs(a.value - b.value) < 1e-6
    for k in a.priors:
        assert abs(a.priors[k] - b.priors[k]) < 1e-6


# --------------------------------------------------------------------------- #
# value_scale: post-multiplier on the network's value output. Pass the
# training-time value_scale (the simulator's reward cap) to recover raw units.
# --------------------------------------------------------------------------- #

def test_default_value_scale_is_raw_passthrough():
    """Pin: default ``value_scale`` is ``DEFAULT_VALUE_SCALE`` (= 1.0), so the
    network's tanh-bounded ``(-1, +1)`` value is returned unchanged."""
    from alpha_rule.evaluation.neural_evaluator import DEFAULT_VALUE_SCALE
    evaluator, _ = _setup()
    assert evaluator.value_scale == DEFAULT_VALUE_SCALE == 1.0
    res = evaluator.evaluate(_root())
    # Capture the raw network output.
    import torch as _t
    ids = evaluator.model.tokenizer.encode(_root().name, max_len=evaluator.max_len).unsqueeze(0)
    evaluator.model.eval()
    with _t.no_grad():
        _, raw = evaluator.model(ids)
    assert abs(res.value - float(raw.item())) < 1e-4


def test_value_scale_explicit_one_preserves_raw_output():
    """Passing ``value_scale=1.0`` opts into raw-output passthrough."""
    import torch as _t
    torch.manual_seed(11)
    g = AllenIntervalGrammar(event_types=("A", "B"), relations=("<",))
    tok = GrammarTokenizer(g)
    model = AllenFormulaNet(tok, d_model=16, nhead=2, num_layers=1, max_len=12)
    evaluator = NeuralEvaluator(model, g, max_len=12, value_scale=1.0)
    res = evaluator.evaluate(_root())
    ids = tok.encode(_root().name, max_len=12).unsqueeze(0)
    model.eval()
    with _t.no_grad():
        _, raw = model(ids)
    assert abs(res.value - float(raw.item())) < 1e-6


def test_value_scale_multiplies_value_output():
    """``value_scale=100`` returns 100 × raw_network_output (relative to a
    raw-passthrough evaluator on the same model)."""
    torch.manual_seed(11)
    g = AllenIntervalGrammar(event_types=("A", "B"), relations=("<",))
    tok = GrammarTokenizer(g)
    model = AllenFormulaNet(tok, d_model=16, nhead=2, num_layers=1, max_len=12)
    eval_raw    = NeuralEvaluator(model, g, max_len=12, value_scale=1.0)
    eval_scaled = NeuralEvaluator(model, g, max_len=12, value_scale=100.0)
    r_raw    = eval_raw.evaluate(_root())
    r_scaled = eval_scaled.evaluate(_root())
    # Priors unaffected.
    for k in r_raw.priors:
        assert abs(r_raw.priors[k] - r_scaled.priors[k]) < 1e-6
    # Value is 100× the raw network output.
    assert abs(r_scaled.value - 100.0 * r_raw.value) < 1e-5


def test_value_scale_rejects_nonpositive():
    import pytest
    g = AllenIntervalGrammar(event_types=("A", "B"), relations=("<",))
    tok = GrammarTokenizer(g)
    model = AllenFormulaNet(tok, d_model=16, nhead=2, num_layers=1, max_len=12)
    with pytest.raises(ValueError):
        NeuralEvaluator(model, g, max_len=12, value_scale=0.0)
    with pytest.raises(ValueError):
        NeuralEvaluator(model, g, max_len=12, value_scale=-2.5)
