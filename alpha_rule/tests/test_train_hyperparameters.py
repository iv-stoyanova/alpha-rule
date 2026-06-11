"""
Tests that ``train()`` exposes every hyperparameter — no defaults are set
only inside called code (model / optimiser / buffer / strategies). Each
new kwarg has a default (so it need not be set) and threads through.
"""
from __future__ import annotations

import pytest

from alpha_rule.evaluation.evaluator import EvalResult
from alpha_rule.grammar.allen import AllenIntervalGrammar
from alpha_rule.mcts.backprop import MaxRewardBackup
from alpha_rule.mcts.selection import PUCTSelection
from alpha_rule.training import train


class _ConstSim:
    def evaluate(self, node):
        return EvalResult(value=1.0)


def _grammar():
    return AllenIntervalGrammar(event_types=("A", "B"), relations=("<",))


def _tiny(**overrides):
    """A fast 2-iteration train with a tiny model; overrides merged in."""
    kwargs = dict(
        grammar=_grammar(), expensive_simulator=_ConstSim(),
        n_iterations=2, n_simulations=4, depth_limit=2,
        buffer_warmup=1, batch_size=4, train_steps_per_iteration=1,
        d_model=16, nhead=2, num_layers=1, max_len=12, seed=0, device="cpu",
    )
    kwargs.update(overrides)
    return train(**kwargs)


def test_train_accepts_every_exposed_hyperparameter():
    """All newly-surfaced knobs set at once — runs without error."""
    log = _tiny(
        temperature=0.8,
        value_scale=50.0,
        dim_feedforward=32, dropout=0.1,
        weight_decay=1e-4, grad_clip=1.0, value_weight=2.0, policy_weight=0.5,
        c_puct=2.0, fpu_reduction=0.1, q_source="filtered_mean",
        backup="percentile", percentile=30.0, min_samples=3,
        dirichlet_eps=0.25, dirichlet_alpha=0.5,
        leaf_eval_mode="simulator", n_chosen_evals=2,
    )
    assert len(log.iterations) == 2
    assert log.model is not None


def test_network_arch_kwargs_propagate_to_model():
    log = _tiny(d_model=24, dim_feedforward=48)
    enc = log.model.encoder
    assert enc.token_embed.embedding_dim == 24                 # d_model
    assert enc.trunk.layers[0].linear1.out_features == 48      # dim_feedforward


def test_backup_string_max_and_percentile_both_run():
    assert len(_tiny(backup="max").iterations) == 2
    assert len(_tiny(backup="percentile", q_source="filtered_mean").iterations) == 2


def test_explicit_strategy_objects_still_accepted():
    log = _tiny(backup=MaxRewardBackup(), selection=PUCTSelection(c_puct=2.0))
    assert len(log.iterations) == 2


def test_invalid_backup_string_raises():
    with pytest.raises(ValueError):
        _tiny(backup="nonsense")


def test_value_scale_recorded_on_log():
    assert _tiny(value_scale=42.0).value_scale == 42.0
