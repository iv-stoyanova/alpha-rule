"""
Tests for read-time reward normalization (``RewardNormalizer``) and the
per-sample seeding fix.

Covers:
    - the normalizer math (Welford std, warm-up, recentering, passthrough),
    - ``normalize``/``denormalize`` are exact inverses (the "consistent by
      construction" guarantee between the value target and the NN de-scale),
    - ``PUCTSelection`` normalizes the visited Q and the FPU parent Q, and is
      byte-identical when ``normalizer=None``,
    - ``Trajectory.value_targets`` recentered single-scale path, and that
      ``norm_*=None`` reproduces the asymmetric mapping,
    - the seed-aware ``_multi_sample_chosen_reward`` (independent samples; the
      ``base_seed=None`` path unchanged; warn-once when a simulator ignores the
      seed), and ``RuleSimulator.evaluate(seed=...)`` re-seeding env + builder.
"""
import math
import warnings

import pytest

from alpha_rule.mcts.normalize import RewardNormalizer
from alpha_rule.mcts.replay import Trajectory, TrajectoryStep
from alpha_rule.mcts.selection import PUCTSelection
from alpha_rule.mcts.self_play import _multi_sample_chosen_reward


# --------------------------------------------------------------------------- #
# RewardNormalizer math
# --------------------------------------------------------------------------- #

def _norm_with(values):
    n = RewardNormalizer()
    for v in values:
        n.update(v)
    return n


def test_population_std_matches():
    # [-5, 5]: mean 0, population std sqrt((25+25)/2) = 5.
    n = _norm_with([-5.0, 5.0])
    assert math.isclose(n.mean, 0.0, abs_tol=1e-12)
    assert math.isclose(n.std, 5.0, rel_tol=1e-9)


def test_warmup_std_is_one_below_two_samples():
    assert RewardNormalizer().std == 1.0          # count 0
    assert _norm_with([3.0]).std == 1.0           # count 1


def test_skips_non_finite():
    n = _norm_with([1.0, float("inf"), float("-inf"), float("nan"), 3.0])
    assert n.count == 2                            # only 1.0 and 3.0 counted
    assert math.isclose(n.mean, 2.0, rel_tol=1e-9)


def test_normalize_recentered_and_clipped():
    n = _norm_with([-5.0, 5.0])                    # mean 0, std 5, k default 2 -> scale 10
    assert math.isclose(n.normalize(0.0), 0.0)     # the mean maps to 0
    assert math.isclose(n.normalize(5.0), 0.5)
    assert n.normalize(100.0) == 1.0               # clipped
    assert n.normalize(-100.0) == -1.0             # clipped


def test_normalize_recenters_on_nonzero_mean():
    n = _norm_with([10.0, 12.0])                   # mean 11, std 1, scale 2
    assert math.isclose(n.normalize(11.0), 0.0)    # mean -> 0 (recentered)
    assert math.isclose(n.normalize(12.0), 0.5)


def test_normalize_passthrough():
    n = _norm_with([-5.0, 5.0])
    assert n.normalize(None) is None
    assert n.normalize(float("-inf")) == float("-inf")
    assert n.normalize(float("inf")) == float("inf")
    assert math.isnan(n.normalize(float("nan")))


def test_denormalize_is_exact_inverse():
    n = _norm_with([-3.0, 7.0])                    # mean 2, std 5
    for raw in (-4.0, 0.0, 2.0, 6.5):
        z = n.normalize(raw)                       # in-range, not clipped here
        assert math.isclose(n.denormalize(z), raw, rel_tol=1e-9, abs_tol=1e-9)


def test_reset_clears():
    n = _norm_with([1.0, 2.0, 3.0])
    n.reset()
    assert n.count == 0 and n.std == 1.0 and n.mean == 0.0


# --------------------------------------------------------------------------- #
# PUCTSelection read-time normalization
# --------------------------------------------------------------------------- #

class _Node:
    def __init__(self, *, N=0, Q_max=float("-inf"), Q_sum=0.0, N_passers=0,
                 prior=1.0, is_dead=False, children=None):
        self.N = N
        self.Q_max = Q_max
        self.Q_sum = Q_sum
        self.N_passers = N_passers
        self.prior = prior
        self.is_dead = is_dead
        self.children = children if children is not None else []


def test_score_normalizes_visited_q():
    n = _norm_with([-5.0, 5.0])                    # scale 10
    child = _Node(N=1, Q_max=5.0)
    parent = _Node(children=[child])
    sel = PUCTSelection(c_puct=0.0, q_source="max", normalizer=n)
    # c_puct 0 -> u term 0 -> score is the normalized Q (5/10 = 0.5).
    assert math.isclose(sel.score(parent, child), 0.5, rel_tol=1e-9)


def test_score_raw_when_normalizer_none():
    child = _Node(N=1, Q_max=5.0)
    parent = _Node(children=[child])
    sel = PUCTSelection(c_puct=0.0, q_source="max")          # no normalizer
    assert math.isclose(sel.score(parent, child), 5.0, rel_tol=1e-9)


def test_fpu_parent_q_normalized():
    n = _norm_with([-5.0, 5.0])                    # scale 10
    child = _Node(N=0)                             # unvisited -> FPU branch
    parent = _Node(N=1, Q_max=8.0, children=[child])
    sel = PUCTSelection(c_puct=0.0, fpu_reduction=0.0, q_source="max",
                        normalizer=n)
    # parent_q 8 -> normalized 0.8, no reduction, fpu_baseline -inf, u 0.
    assert math.isclose(sel.score(parent, child), 0.8, rel_tol=1e-9)


def test_fpu_baseline_zero_is_the_mean():
    n = _norm_with([-5.0, 5.0])
    child = _Node(N=0)
    # parent dragged below the mean: normalized parent_q is negative.
    parent = _Node(N=1, Q_max=-6.0, children=[child])
    sel = PUCTSelection(c_puct=0.0, fpu_reduction=0.0, q_source="max",
                        fpu_baseline=0.0, normalizer=n)
    # normalized parent_q = -0.6, floored at 0.0 (the mean).
    assert math.isclose(sel.score(parent, child), 0.0, abs_tol=1e-9)


# --------------------------------------------------------------------------- #
# Trajectory.value_targets recentered single scale
# --------------------------------------------------------------------------- #

def _traj(state_value, *, norm=True):
    step = TrajectoryStep(state="s", visit_pi={}, reward=0.0,
                          state_value=state_value)
    if norm:
        return Trajectory(steps=[step], norm_mean=0.0, norm_std=5.0, norm_k=2.0)
    return Trajectory(steps=[step], value_scale=3.0, neg_value_scale=30.0)


def test_value_targets_normalized_single_scale():
    assert math.isclose(_traj(5.0).value_targets()[0], 0.5, rel_tol=1e-9)
    assert _traj(100.0).value_targets()[0] == 1.0          # clipped
    assert _traj(-100.0).value_targets()[0] == -1.0        # clipped, symmetric
    assert math.isclose(_traj(-5.0).value_targets()[0], -0.5, rel_tol=1e-9)


def test_value_targets_normalized_nonfinite_floor():
    t = Trajectory(
        steps=[TrajectoryStep(state="s", visit_pi={}, reward=float("-inf"),
                              state_value=float("-inf"))],
        norm_mean=0.0, norm_std=5.0, norm_k=2.0,
    )
    assert t.value_targets()[0] == -1.0


def test_value_targets_asymmetric_when_norm_off():
    # norm_*=None -> the historical asymmetric pos/neg mapping.
    assert math.isclose(_traj(3.0, norm=False).value_targets()[0], 1.0)   # 3/3
    assert math.isclose(_traj(-30.0, norm=False).value_targets()[0], -1.0)  # -30/30


def test_value_targets_normalize_matches_denormalize():
    # value_targets and denormalize are inverses under the same stats.
    n = _norm_with([-3.0, 7.0])
    raw = 4.5
    z = _traj_value_via_norm(n, raw)
    assert math.isclose(n.denormalize(z), raw, rel_tol=1e-9, abs_tol=1e-9)


def _traj_value_via_norm(n: RewardNormalizer, raw: float) -> float:
    t = Trajectory(steps=[TrajectoryStep(state="s", visit_pi={}, reward=0.0,
                                         state_value=raw)],
                   norm_mean=n.mean, norm_std=n.std, norm_k=2.0)
    return t.value_targets()[0]


# --------------------------------------------------------------------------- #
# Seed-aware multi-sampling
# --------------------------------------------------------------------------- #

class _SeedSim:
    """evaluate returns float(seed) when seeded, else a constant."""
    def __init__(self, const=0.0):
        self.const = const

    def evaluate(self, node, *, seed=None):
        return self.const if seed is None else float(seed)


class _NoSeedSim:
    """A simulator whose evaluate does not accept a seed kwarg."""
    def __init__(self, const=0.7):
        self.const = const

    def evaluate(self, node):
        return self.const


def test_multi_sample_base_seed_varies_samples():
    sim = _SeedSim()
    # base_seed=100, n=3 -> seeds 100,101,102 -> mean 101.
    out = _multi_sample_chosen_reward(sim, object(), 3, base_seed=100)
    assert math.isclose(out, 101.0, rel_tol=1e-9)


def test_multi_sample_base_seed_none_is_unseeded():
    sim = _SeedSim(const=0.42)
    out = _multi_sample_chosen_reward(sim, object(), 3, base_seed=None)
    assert math.isclose(out, 0.42, rel_tol=1e-9)


def test_multi_sample_warns_once_when_seed_ignored():
    sim = _NoSeedSim(const=0.7)
    warn_state = {"warned": False}
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = _multi_sample_chosen_reward(
            sim, object(), 3, base_seed=5, warn_state=warn_state)
    assert math.isclose(out, 0.7, rel_tol=1e-9)        # graceful fallback
    assert sum("seed" in str(x.message) for x in w) == 1


# --------------------------------------------------------------------------- #
# RuleSimulator.evaluate seed override (env + builder)
# --------------------------------------------------------------------------- #

class _RecordEnv:
    def __init__(self):
        self.seeds = []

    def reset(self, seed=None):
        self.seeds.append(seed)
        return None, {}


def test_rule_simulator_seed_overrides_env_and_builder():
    pytest.importorskip("gymnasium")
    from alpha_rule.evaluation.rule_simulator import RuleSimulator

    builder_seeds = []

    def builder(env, **kwargs):
        builder_seeds.append(kwargs.get("seed"))
        return ("qtable", lambda s: 0)

    sim = RuleSimulator(
        "dummy-env", builder, lambda e, r: e, lambda agent, env: 1.0,
        reward_scale=1.0, seed=123, agent_builder_kwargs={"seed": 123},
    )
    sim._env = _RecordEnv()                 # skip gym.make

    class _N:
        name = "A"

    sim.evaluate(_N(), seed=7)
    assert sim._env.seeds[-1] == 7          # env re-seeded with the per-call seed
    assert builder_seeds[-1] == 7           # builder seed overridden

    sim.evaluate(_N())                      # no per-call seed -> instance seed
    assert sim._env.seeds[-1] == 123
    assert builder_seeds[-1] == 123
