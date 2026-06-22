"""
Tests for value-sample harvesting from the search tree.

Covers:
    - ``TreeQmaxCollector`` (min_visits filter, non-finite skip, dead-subtree
      skip, root excluded),
    - ``ValueBuffer`` push/sample/len,
    - ``Trajectory.value_sample_targets`` normalization matches ``value_targets``,
    - ``train_value_step`` trains the value head and leaves the policy-head
      params untouched,
    - ``run_self_play`` harvests samples when given a collector,
    - ``train`` runs value-only steps when a collector + ``value_train_steps`` are
      set, and none otherwise.
"""
from __future__ import annotations

import math

import pytest

from alpha_rule.mcts.replay import Trajectory, TrajectoryStep, ValueBuffer
from alpha_rule.mcts.value_collect import TreeQmaxCollector


# --------------------------------------------------------------------------- #
# TreeQmaxCollector
# --------------------------------------------------------------------------- #

class _N:
    def __init__(self, name, *, N=0, Q_max=float("-inf"), is_dead=False, children=None):
        self.name = name
        self.N = N
        self.Q_max = Q_max
        self.is_dead = is_dead
        self.children = children or []


def _tree():
    aa = _N("AA", N=3, Q_max=2.0)
    ab = _N("AB", N=1, Q_max=0.5)                  # below min_visits
    a = _N("A", N=5, Q_max=1.0, children=[aa, ab])
    b = _N("B", N=2, Q_max=float("-inf"))          # finite check fails
    ca = _N("CA", N=10, Q_max=9.0)                 # under a dead node
    c = _N("C", N=4, Q_max=-3.0, is_dead=True, children=[ca])
    return _N("R", N=11, Q_max=2.0, children=[a, b, c])


def test_collector_filters_and_excludes_root():
    out = dict(TreeQmaxCollector(min_visits=2).collect(_tree()))
    assert out == {"A": 1.0, "AA": 2.0}            # AB<min, B non-finite, C/CA dead, R excluded


def test_collector_min_visits_one_keeps_more():
    out = dict(TreeQmaxCollector(min_visits=1).collect(_tree()))
    assert out == {"A": 1.0, "AA": 2.0, "AB": 0.5}  # still skips non-finite B and dead C/CA


def test_collector_rejects_bad_min_visits():
    with pytest.raises(ValueError):
        TreeQmaxCollector(min_visits=0)


# --------------------------------------------------------------------------- #
# ValueBuffer
# --------------------------------------------------------------------------- #

def test_value_buffer_push_and_sample():
    vb = ValueBuffer(capacity=10)
    vb.push([("a", 0.5), ("b", -0.3)])
    assert len(vb) == 2
    assert len(vb.sample(1)) == 1
    assert len(vb.sample(5)) == 2                  # capped at occupancy


def test_value_buffer_evicts_oldest():
    vb = ValueBuffer(capacity=2)
    vb.push([("a", 0.1), ("b", 0.2), ("c", 0.3)])
    assert len(vb) == 2                            # FIFO eviction


# --------------------------------------------------------------------------- #
# Trajectory.value_sample_targets matches value_targets
# --------------------------------------------------------------------------- #

def test_value_sample_targets_match_value_targets_normalized():
    raws = [5.0, -5.0, 100.0]
    steps = [
        TrajectoryStep(state=f"s{i}", visit_pi={}, reward=0.0, state_value=r)
        for i, r in enumerate(raws)
    ]
    t = Trajectory(steps=steps, norm_mean=0.0, norm_std=5.0, norm_k=2.0,
                   value_samples=[(f"n{i}", r) for i, r in enumerate(raws)])
    vt = t.value_targets()
    vst = [z for _, z in t.value_sample_targets()]
    assert vst == vt


def test_value_sample_targets_asymmetric_when_norm_off():
    t = Trajectory(steps=[], value_scale=3.0, neg_value_scale=30.0,
                   value_samples=[("a", 3.0), ("b", -30.0), ("c", -60.0)])
    out = dict(t.value_sample_targets())
    assert out["a"] == 1.0          # 3/3
    assert out["b"] == -1.0         # -30/30
    assert out["c"] == -1.0         # clipped


def test_value_sample_targets_nonfinite_floor():
    t = Trajectory(steps=[], norm_mean=0.0, norm_std=5.0, norm_k=2.0,
                   value_samples=[("a", float("-inf"))])
    assert t.value_sample_targets() == [("a", -1.0)]


# --------------------------------------------------------------------------- #
# train_value_step: trains the value head, leaves the policy head params alone
# --------------------------------------------------------------------------- #

def _model(events=("A", "B")):
    import torch
    from alpha_rule.grammar.allen import AllenIntervalGrammar
    from alpha_rule.nn.model import AllenFormulaNet
    from alpha_rule.nn.tokenizer import GrammarTokenizer

    torch.manual_seed(0)
    g = AllenIntervalGrammar(event_types=events, relations=("<",))
    return AllenFormulaNet(GrammarTokenizer(g), d_model=16, nhead=2,
                           num_layers=1, max_len=12)


def test_train_value_step_trains_value_not_policy():
    torch = pytest.importorskip("torch")
    from alpha_rule.nn.training import train_value_step

    model = _model()
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    pol_before = model.policy.linear.weight.detach().clone()
    val_before = model.value.linear.weight.detach().clone()

    batch = [("A", 0.6), ("A B", 0.6)]
    losses = [train_value_step(model, opt, batch, max_len=12) for _ in range(25)]

    assert torch.equal(model.policy.linear.weight, pol_before)       # policy head untouched
    assert not torch.equal(model.value.linear.weight, val_before)    # value head moved
    assert losses[-1] < losses[0]                                    # value loss decreased


def test_train_value_step_empty_batch_is_noop():
    pytest.importorskip("torch")
    import torch
    from alpha_rule.nn.training import train_value_step
    model = _model()
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    assert train_value_step(model, opt, [], max_len=12) == 0.0


# --------------------------------------------------------------------------- #
# Integration: run_self_play harvest + train value-only steps
# --------------------------------------------------------------------------- #

class _ConstSim:
    def __init__(self, value=1.0):
        self.value = value

    def evaluate(self, node):
        from alpha_rule.evaluation.evaluator import EvalResult
        return EvalResult(value=self.value)


def test_run_self_play_harvests_more_than_committed_steps():
    pytest.importorskip("numpy")
    import numpy as np
    from alpha_rule.grammar.allen import AllenIntervalGrammar
    from alpha_rule.mcts.self_play import run_self_play

    g = AllenIntervalGrammar(event_types=("A", "B", "C"), relations=("<",))
    traj = run_self_play(
        grammar=g, simulator=_ConstSim(1.0), network_evaluator=None,
        n_simulations=20, depth_limit=2, leaf_eval_mode="simulator",
        value_sample_collector=TreeQmaxCollector(min_visits=1),
        rng=np.random.default_rng(0),
    )
    assert len(traj.value_samples) > len(traj.steps)


def test_run_self_play_no_collector_no_harvest():
    pytest.importorskip("numpy")
    import numpy as np
    from alpha_rule.grammar.allen import AllenIntervalGrammar
    from alpha_rule.mcts.self_play import run_self_play

    g = AllenIntervalGrammar(event_types=("A", "B"), relations=("<",))
    traj = run_self_play(
        grammar=g, simulator=_ConstSim(1.0), network_evaluator=None,
        n_simulations=8, depth_limit=2, leaf_eval_mode="simulator",
        rng=np.random.default_rng(0),
    )
    assert traj.value_samples == []


def _tiny_train(**over):
    pytest.importorskip("torch")
    from alpha_rule.grammar.allen import AllenIntervalGrammar
    from alpha_rule.training import train
    kw = dict(
        grammar=AllenIntervalGrammar(event_types=("A", "B"), relations=("<",)),
        expensive_simulator=_ConstSim(1.0),
        n_iterations=2, n_simulations=4, depth_limit=2,
        buffer_warmup=1, batch_size=4, train_steps_per_iteration=1,
        d_model=16, nhead=2, num_layers=1, max_len=12, seed=0, device="cpu",
    )
    kw.update(over)
    return train(**kw)


def test_train_runs_value_only_steps_with_collector(monkeypatch):
    import alpha_rule.nn.training as nn_training
    calls = {"n": 0}
    real = nn_training.train_value_step

    def spy(*a, **k):
        calls["n"] += 1
        return real(*a, **k)

    monkeypatch.setattr(nn_training, "train_value_step", spy)
    _tiny_train(value_sample_collector=TreeQmaxCollector(min_visits=1),
                value_train_steps=2)
    assert calls["n"] > 0


def test_train_no_value_steps_without_collector(monkeypatch):
    import alpha_rule.nn.training as nn_training
    calls = {"n": 0}
    real = nn_training.train_value_step

    def spy(*a, **k):
        calls["n"] += 1
        return real(*a, **k)

    monkeypatch.setattr(nn_training, "train_value_step", spy)
    _tiny_train()                                  # no collector, value_train_steps=0
    assert calls["n"] == 0


# --------------------------------------------------------------------------- #
# Visibility: harvest metrics on the iteration log + CSV columns
# --------------------------------------------------------------------------- #

def test_iteration_log_records_harvest_metrics():
    log = _tiny_train(value_sample_collector=TreeQmaxCollector(min_visits=1),
                      value_train_steps=2)
    assert any(it.n_value_samples > 0 for it in log.iterations)        # harvested
    assert all(it.value_harvest_loss >= 0.0 for it in log.iterations)  # recorded


def test_iteration_log_no_harvest_without_collector():
    log = _tiny_train()
    assert all(it.n_value_samples == 0 for it in log.iterations)


def test_csv_columns_include_harvest_metrics():
    from alpha_rule.training.csv_logger import CSV_COLUMNS
    assert "n_value_samples" in CSV_COLUMNS
    assert "value_harvest_loss" in CSV_COLUMNS
