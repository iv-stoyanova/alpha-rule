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


# --------------------------------------------------------------------------- #
# from_simulator: wire value_scale to the simulator's reward_scale so the NN
# value and the simulator reward share one scale in the MCTS backup.
# --------------------------------------------------------------------------- #

def test_from_simulator_matches_reward_scale():
    g = AllenIntervalGrammar(event_types=("A", "B"), relations=("<",))
    tok = GrammarTokenizer(g)
    model = AllenFormulaNet(tok, d_model=16, nhead=2, num_layers=1, max_len=12)

    class _Sim:
        reward_scale = 4.0
        def evaluate(self, node):
            return 2.0

    ev = NeuralEvaluator.from_simulator(model, g, _Sim(), max_len=12)
    assert ev.value_scale == 4.0

    class _PlainSim:                                  # no reward_scale -> 1.0
        def evaluate(self, node):
            return 1.0

    assert NeuralEvaluator.from_simulator(model, g, _PlainSim(), max_len=12).value_scale == 1.0


# --------------------------------------------------------------------------- #
# end_prior_scale: down-weight the terminal (<END>) prior so it stops dominating
# the softmax. <END> is only applicable at non-root nodes.
# --------------------------------------------------------------------------- #

def _child_with_end():
    """A non-root node (one event applied) where END_RULE is applicable."""
    torch.manual_seed(11)
    g = AllenIntervalGrammar(event_types=("A", "B"), relations=("<",))
    tok = GrammarTokenizer(g)
    model = AllenFormulaNet(tok, d_model=16, nhead=2, num_layers=1, max_len=12)
    root = g.root()
    child = g.apply(root, g.applicable_productions(root)[0])
    assert "END_RULE" in {p.name for p in g.applicable_productions(child)}
    return model, g, child


def test_end_prior_scale_downweights_end_and_renormalizes():
    model, g, child = _child_with_end()
    base = NeuralEvaluator(model, g, max_len=12, end_prior_scale=1.0)
    down = NeuralEvaluator(model, g, max_len=12, end_prior_scale=0.3)
    rb = base.evaluate(child)
    rd = down.evaluate(child)
    assert rd.priors["END_RULE"] < rb.priors["END_RULE"]        # END falls
    non_end = [k for k in rb.priors if k != "END_RULE"]
    assert all(rd.priors[k] > rb.priors[k] for k in non_end)    # events rise
    assert abs(sum(rd.priors.values()) - 1.0) < 1e-5            # still a distribution


def test_end_prior_scale_one_is_noop():
    model, g, child = _child_with_end()
    base = NeuralEvaluator(model, g, max_len=12)                # default 1.0
    one = NeuralEvaluator(model, g, max_len=12, end_prior_scale=1.0)
    rb = base.evaluate(child)
    ro = one.evaluate(child)
    for k in rb.priors:
        assert abs(rb.priors[k] - ro.priors[k]) < 1e-9


def test_end_prior_scale_rejects_negative():
    import pytest
    g = AllenIntervalGrammar(event_types=("A", "B"), relations=("<",))
    tok = GrammarTokenizer(g)
    model = AllenFormulaNet(tok, d_model=16, nhead=2, num_layers=1, max_len=12)
    with pytest.raises(ValueError):
        NeuralEvaluator(model, g, max_len=12, end_prior_scale=-0.1)


def test_neural_evaluator_drives_self_play_end_to_end():
    """The NN evaluator plugs into run_self_play (leaf_eval_mode='nn'); with
    from_simulator wiring the scale, the stored value targets stay in [-1, 1]."""
    import numpy as np

    from alpha_rule.mcts.replay import ReplayBuffer
    from alpha_rule.mcts.self_play import run_self_play

    torch.manual_seed(3)
    g = AllenIntervalGrammar(event_types=("A", "B"), relations=("<",))
    tok = GrammarTokenizer(g)
    model = AllenFormulaNet(tok, d_model=16, nhead=2, num_layers=1, max_len=32)

    class _Sim:
        reward_scale = 4.0
        def evaluate(self, node):
            return 2.0

    sim = _Sim()
    ev = NeuralEvaluator.from_simulator(model, g, sim, max_len=32)
    traj = run_self_play(
        grammar=g, simulator=sim, network_evaluator=ev,
        n_simulations=8, depth_limit=3, leaf_eval_mode="nn",
        rng=np.random.default_rng(0),
    )
    assert len(traj.steps) >= 1
    assert traj.value_scale == 4.0
    buf = ReplayBuffer(capacity=20)
    buf.push_trajectory(traj)
    for row in buf._buf:
        assert -1.0 <= row[2] <= 1.0
