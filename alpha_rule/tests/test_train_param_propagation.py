"""
Tests that ``train`` propagates its parameters into the components it wires,
and that the new search-health logs / risky-config warnings behave.

Pins:
    - ``value_scale=None`` inherits the simulator's ``reward_scale`` and the
      resolved scale lands on ``TrainingLog.value_scale``.
    - ``max_len`` reaches both the model and the log.
    - The policy target stored in a step is the tau=1 visit distribution,
      decoupled from the action-sampling temperature.
    - ``train``'s defaults encode the agreed design choices (temperature
      decay on, Dirichlet exploration on).
    - ``_warn_risky_config`` fires exactly on the known footguns and stays
      silent on a clean config.
    - ``n_dead_rules`` is cumulative/monotonic and ``buffer_fill_fraction``
      stays in [0, 1] and reaches the logger CSV.
"""
from __future__ import annotations

import csv
import inspect
import warnings

import numpy as np

from alpha_rule.evaluation.evaluator import EvalResult
from alpha_rule.grammar.allen import AllenIntervalGrammar
from alpha_rule.mcts.self_play import run_self_play
from alpha_rule.training import train
from alpha_rule.training.train import _warn_risky_config


class _ConstSim:
    """Finite reward for every rule; optional reward_scale."""
    def __init__(self, value=1.0, reward_scale=None):
        self.value = value
        if reward_scale is not None:
            self.reward_scale = reward_scale

    def evaluate(self, node):
        return EvalResult(value=self.value)


def _grammar(events=("A", "B")):
    return AllenIntervalGrammar(event_types=events, relations=("<",))


def _tiny(**overrides):
    kwargs = dict(
        grammar=_grammar(), expensive_simulator=_ConstSim(),
        n_iterations=3, n_simulations=4, depth_limit=2,
        buffer_warmup=1, batch_size=4, train_steps_per_iteration=1,
        d_model=16, nhead=2, num_layers=1, max_len=12, seed=0, device="cpu",
    )
    kwargs.update(overrides)
    return train(**kwargs)


# --------------------------------------------------------------------------- #
# Scalar propagation
# --------------------------------------------------------------------------- #

def test_value_scale_none_inherits_simulator_reward_scale():
    log = _tiny(expensive_simulator=_ConstSim(reward_scale=3.0), value_scale=None)
    assert log.value_scale == 3.0


def test_max_len_propagates_to_model_and_log():
    log = _tiny(max_len=20)
    assert log.max_len == 20
    assert log.model.max_len == 20
    # Position embedding capacity matches the requested budget.
    assert log.model.encoder.pos_embed.num_embeddings == 20


# --------------------------------------------------------------------------- #
# Temperature decoupling: stored policy target is tau=1, not the sampling temp
# --------------------------------------------------------------------------- #

def test_policy_target_decoupled_from_sampling_temperature():
    """With depth_limit=1 the search tree is identical for any sampling
    temperature (temperature only affects the action drawn *after* the tree is
    built). So the stored visit_pi must be byte-identical for temperature=0 and
    temperature=1 -- which is only true when the target is the tau=1
    distribution rather than the (one-hot at tau=0) sampling distribution."""
    g = _grammar(("A", "B", "C"))
    sim = _ConstSim(1.0)

    def _go(temp):
        return run_self_play(
            grammar=g, simulator=sim, network_evaluator=None,
            n_simulations=12, depth_limit=1, temperature=temp,
            rng=np.random.default_rng(0),
        )

    greedy = _go(0.0)
    proportional = _go(1.0)
    pi_greedy = greedy.steps[0].visit_pi
    pi_prop = proportional.steps[0].visit_pi
    # More than one branch was visited, so a one-hot target would differ.
    assert len([v for v in pi_prop.values() if v > 0]) >= 2
    assert pi_greedy == pi_prop                        # both are the tau=1 target
    assert abs(sum(pi_greedy.values()) - 1.0) < 1e-9


# --------------------------------------------------------------------------- #
# Agreed-on defaults
# --------------------------------------------------------------------------- #

def test_train_defaults_match_design_choices():
    params = inspect.signature(train).parameters
    assert params["temperature_final"].default == 0.1     # decay on
    assert params["dirichlet_eps"].default == 0.25        # exploration on


# --------------------------------------------------------------------------- #
# Risky-config warnings
# --------------------------------------------------------------------------- #

def _warns(**kwargs):
    """Run _warn_risky_config and return the list of warning messages."""
    base = dict(
        max_len=64, depth_limit=5, explicit_value_scale=None,
        simulator=_ConstSim(), backup="max", selection=None,
        q_source="max", eval_simulator=None, eval_every=5,
    )
    base.update(kwargs)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _warn_risky_config(**base)
    return [str(w.message) for w in caught]


def test_warns_on_max_len_too_small_for_depth():
    msgs = _warns(max_len=6, depth_limit=10)
    assert any("max_len" in m for m in msgs)


def test_warns_on_value_scale_mismatch():
    msgs = _warns(explicit_value_scale=10.0, simulator=_ConstSim(reward_scale=3.0))
    assert any("value_scale" in m and "reward_scale" in m for m in msgs)


def test_no_warn_when_value_scale_matches_reward_scale():
    msgs = _warns(explicit_value_scale=3.0, simulator=_ConstSim(reward_scale=3.0))
    assert not any("value_scale" in m for m in msgs)


def test_warns_on_percentile_backup_with_wrong_q_source():
    msgs = _warns(backup="percentile", selection=None, q_source="max")
    assert any("percentile" in m and "filtered_mean" in m for m in msgs)


def test_no_warn_on_percentile_backup_with_filtered_mean():
    msgs = _warns(backup="percentile", selection=None, q_source="filtered_mean")
    assert not any("filtered_mean" in m for m in msgs)


def test_warns_on_nonpositive_eval_every_with_eval_simulator():
    msgs = _warns(eval_simulator=_ConstSim(), eval_every=0)
    assert any("eval_every" in m for m in msgs)


def test_clean_config_emits_no_warnings():
    assert _warns() == []


# --------------------------------------------------------------------------- #
# Search-depth cap: the search never builds rules deeper than depth_limit
# --------------------------------------------------------------------------- #

def test_run_one_round_respects_depth_limit():
    from alpha_rule.mcts.backprop import MaxRewardBackup
    from alpha_rule.mcts.expansion import RuleExpansion
    from alpha_rule.mcts.selection import PUCTSelection
    from alpha_rule.mcts.self_play import _run_one_round

    g = _grammar(("A", "B", "C"))
    root = g.root()
    _run_one_round(
        root, n_simulations=100, simulator=_ConstSim(1.0),
        network_evaluator=None, selection=PUCTSelection(),
        backup=MaxRewardBackup(), expansion=RuleExpansion(g),
        leaf_eval_mode="simulator", depth_limit=2,
    )

    def max_level(n):
        return max([n.level] + [max_level(c) for c in n.children])

    assert max_level(root) <= 2                  # never expanded past the cap


def test_tight_max_len_survives_deep_search():
    """Regression: with the search capped at depth_limit, max_len =
    depth_limit + 2 is sufficient even under many simulations that would
    otherwise grow the tree past depth_limit and overflow the tokenizer."""
    log = _tiny(
        grammar=_grammar(("A", "B", "C")),
        n_iterations=2, n_simulations=200, depth_limit=4, max_len=6,
        temperature_final=None, dirichlet_eps=0.0,
    )
    assert len(log.iterations) == 2


# --------------------------------------------------------------------------- #
# New search-health logs
# --------------------------------------------------------------------------- #

def test_dead_rules_count_is_cumulative_and_buffer_fill_in_range():
    log = _tiny(n_iterations=4)
    dead_counts = [it.n_dead_rules for it in log.iterations]
    # Cumulative set size never shrinks.
    assert dead_counts == sorted(dead_counts)
    for it in log.iterations:
        assert isinstance(it.n_dead_rules, int) and it.n_dead_rules >= 0
        assert 0.0 <= it.buffer_fill_fraction <= 1.0
    # With an all-finite simulator nothing dies.
    assert dead_counts[-1] == 0
    # The buffer accrued rows past warmup.
    assert log.iterations[-1].buffer_fill_fraction > 0.0


def test_new_metrics_round_trip_through_csv_logger():
    import shutil
    import tempfile

    from alpha_rule.training.csv_logger import AlphaZeroCSVLogger, CSV_COLUMNS

    assert "n_dead_rules" in CSV_COLUMNS
    assert "buffer_fill_fraction" in CSV_COLUMNS

    base_dir = tempfile.mkdtemp()
    try:
        logger = AlphaZeroCSVLogger(
            base_dir=base_dir, env_name="Env", activity="test", strategy="PUCT+Max",
        )
        logger.log_iteration(
            iteration=0, trajectory_length=2, best_reward_in_trajectory=0.5,
            n_failed_evaluations=0, policy_loss=0.1, value_loss=0.2, total_loss=0.3,
            n_dead_rules=7, buffer_fill_fraction=0.25,
        )
        with open(logger.csv_path, newline="") as f:
            rows = list(csv.DictReader(f))
    finally:
        shutil.rmtree(base_dir, ignore_errors=True)
    assert rows[0]["n_dead_rules"] == "7"
    assert rows[0]["buffer_fill_fraction"] == "0.25"
