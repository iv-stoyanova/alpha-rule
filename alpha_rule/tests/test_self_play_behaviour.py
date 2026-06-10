"""
End-to-end behaviour tests for self-play: value-target / reward-scaling wiring
and search/trajectory invariants. Complements the unit tests in
test_value_target.py and test_replay_buffer.py.
"""
from __future__ import annotations

import math

import numpy as np

from alpha_rule.evaluation.evaluator import EvalResult
from alpha_rule.grammar.allen import AllenIntervalGrammar
from alpha_rule.mcts.backprop import MaxRewardBackup, PercentileRewardBackup
from alpha_rule.mcts.expansion import RuleExpansion
from alpha_rule.mcts.replay import ReplayBuffer
from alpha_rule.mcts.self_play import _normalised_visit_distribution, run_self_play
from alpha_rule.mcts.value_target import ExpectedValue, MaxValue, RealizedReturn


class _ConstantSimulator:
    def __init__(self, value=1.0):
        self.value = value

    def evaluate(self, node):
        return self.value


class _ScaledSimulator:
    """Constant-reward simulator that also exposes a positive reward cap."""

    def __init__(self, value, reward_scale):
        self.value = value
        self.reward_scale = reward_scale

    def evaluate(self, node):
        return self.value


# --------------------------------------------------------------------------- #
# value_scale auto-read + reward scaling
# --------------------------------------------------------------------------- #

def test_value_scale_autoread_and_small_reward_not_crushed():
    g = AllenIntervalGrammar(event_types=("A", "B"), relations=("<",))

    # Reward above the cap clips to 1.0.
    traj = run_self_play(
        grammar=g, simulator=_ScaledSimulator(10.0, reward_scale=2.0),
        n_simulations=8, depth_limit=1, rng=np.random.default_rng(0),
        leaf_eval_mode="simulator",
    )
    assert traj.value_scale == 2.0
    buf = ReplayBuffer(capacity=10)
    buf.push_trajectory(traj)
    assert list(buf._buf)[0][2] == 1.0                # clip(10 / 2, -1, 1)

    # A small reward is NOT crushed toward 0 (the old /100 scheme gave 0.001).
    traj2 = run_self_play(
        grammar=g, simulator=_ScaledSimulator(0.1, reward_scale=2.0),
        n_simulations=8, depth_limit=1, rng=np.random.default_rng(0),
        leaf_eval_mode="simulator",
    )
    buf2 = ReplayBuffer(capacity=10)
    buf2.push_trajectory(traj2)
    z = list(buf2._buf)[0][2]
    assert abs(z - 0.05) < 1e-9 and z > 0.04          # 0.1 / 2 = 0.05


def test_value_scale_none_when_simulator_has_no_reward_scale():
    g = AllenIntervalGrammar(event_types=("A", "B"), relations=("<",))
    traj = run_self_play(
        grammar=g, simulator=_ConstantSimulator(0.5),
        n_simulations=6, depth_limit=1, rng=np.random.default_rng(0),
        leaf_eval_mode="simulator",
    )
    assert traj.value_scale is None
    buf = ReplayBuffer(capacity=10)
    buf.push_trajectory(traj)
    assert list(buf._buf)[0][2] == 0.5                # scale falls back to 1.0


def test_minus_inf_reward_maps_to_minus_one():
    g = AllenIntervalGrammar(event_types=("A", "B"), relations=("<",))

    class _DeadSim:
        reward_scale = 2.0

        def evaluate(self, node):
            return float("-inf")

    traj = run_self_play(
        grammar=g, simulator=_DeadSim(), n_simulations=20, depth_limit=2,
        rng=np.random.default_rng(0), leaf_eval_mode="simulator",
    )
    # No live continuations -> empty trajectory; nothing to assert on rows, but
    # the value-target mapping itself is pinned in test_replay_buffer. Here we
    # only confirm the all-dead episode is handled (empty, finite scale).
    assert traj.steps == []
    assert traj.value_scale == 2.0


# --------------------------------------------------------------------------- #
# value target <-> policy target agreement, auto-pairing, RealizedReturn
# --------------------------------------------------------------------------- #

def test_dead_branch_agreement_value_and_policy_exclude_same_child():
    g = AllenIntervalGrammar(event_types=("A", "B", "C"), relations=("<",))
    exp = RuleExpansion(g)
    root = g.root()
    a = exp.expand(root)
    b = exp.expand(root)
    c = exp.expand(root)
    a.N, a.Q_sum, a.N_passers = 5, 250.0, 5          # filtered mean 50, live
    c.N, c.Q_sum, c.N_passers = 2, 40.0, 2           # filtered mean 20, live
    b.N, b.Q_sum, b.N_passers, b.is_dead = 3, 60.0, 3, True   # dead

    pi = _normalised_visit_distribution(root, temperature=1.0)
    z = ExpectedValue().state_value(root)

    assert b.parent_action not in pi                 # policy drops the dead child
    assert set(pi) == {a.parent_action, c.parent_action}
    assert abs(z - (5 * 50.0 + 2 * 20.0) / (5 + 2)) < 1e-9   # value over live only


def test_autopair_percentile_backup_uses_expected_value():
    g = AllenIntervalGrammar(event_types=("A", "B"), relations=("<",))

    class _VarSim:                                   # children differ -> Expected != Max
        def evaluate(self, node):
            return float(len(node.name))

    def go(value_target=None):
        return run_self_play(
            grammar=g, simulator=_VarSim(),
            backup=PercentileRewardBackup(min_samples=1),
            value_target=value_target, n_simulations=15, depth_limit=2,
            rng=np.random.default_rng(0), leaf_eval_mode="simulator",
        )

    auto = [s.state_value for s in go().steps]                 # auto-pairs
    explicit = [s.state_value for s in go(ExpectedValue()).steps]
    assert auto == explicit                          # Percentile -> ExpectedValue


def test_realized_return_root_none_falls_back_to_clipped_reward():
    g = AllenIntervalGrammar(event_types=("A", "B"), relations=("<",))
    traj = run_self_play(
        grammar=g, simulator=_ScaledSimulator(3.0, reward_scale=3.0),
        n_simulations=6, depth_limit=2, value_target=RealizedReturn(),
        rng=np.random.default_rng(0), leaf_eval_mode="simulator",
    )
    assert traj.steps[0].state_value is None         # root never stamped
    buf = ReplayBuffer(capacity=10)
    buf.push_trajectory(traj)
    # The root row falls back to its own reward (3.0) clipped/scaled by 3.0 -> 1.0.
    assert list(buf._buf)[0][2] == 1.0
    assert all(-1.0 <= row[2] <= 1.0 for row in buf._buf)


# --------------------------------------------------------------------------- #
# Search / trajectory invariants
# --------------------------------------------------------------------------- #

def test_all_inf_simulator_yields_empty_trajectory():
    g = AllenIntervalGrammar(event_types=("A", "B"), relations=("<",))

    class _DeadSim:
        def __init__(self):
            self.calls = 0

        def evaluate(self, node):
            self.calls += 1
            return float("-inf")

    sim = _DeadSim()
    traj = run_self_play(
        grammar=g, simulator=sim, n_simulations=30, depth_limit=4,
        rng=np.random.default_rng(0), leaf_eval_mode="simulator",
    )
    assert traj.steps == []
    # Only the two root children are evaluated before the cascade kills the root;
    # after that the dead short-circuit spends no further simulator calls.
    assert sim.calls <= 2


def test_leaf_eval_mode_nn_never_values_a_terminal_with_the_network():
    g = AllenIntervalGrammar(event_types=("A",), relations=("<",))

    class _Net:
        def __init__(self):
            self.seen = []

        def evaluate(self, node):
            self.seen.append(node.name)
            return EvalResult(value=0.0, priors={})

    class _RecSim(_ConstantSimulator):
        def __init__(self):
            super().__init__(1.0)
            self.names = []

        def evaluate(self, node):
            self.names.append(node.name)
            return super().evaluate(node)

    net = _Net()
    sim = _RecSim()
    run_self_play(
        grammar=g, simulator=sim, network_evaluator=net,
        n_simulations=20, depth_limit=3, rng=np.random.default_rng(0),
        leaf_eval_mode="nn",
    )
    # The network is used for priors + non-terminal leaf values, but never to
    # value a terminal node; terminals are valued by the simulator.
    assert not any(name.endswith("<END>") for name in net.seen)
    assert any(name.endswith("<END>") for name in sim.names)


def test_reproducibility_includes_state_value_and_targets():
    g = AllenIntervalGrammar(event_types=("A", "B"), relations=("<",))

    def go():
        return run_self_play(
            grammar=g, simulator=_ConstantSimulator(1.0),
            n_simulations=8, depth_limit=3, rng=np.random.default_rng(123),
            leaf_eval_mode="simulator",
        )

    a, b = go(), go()
    assert [s.state for s in a.steps] == [s.state for s in b.steps]
    assert [s.visit_pi for s in a.steps] == [s.visit_pi for s in b.steps]
    assert [s.state_value for s in a.steps] == [s.state_value for s in b.steps]
    assert a.value_targets() == b.value_targets()


def test_temperature_zero_is_argmax():
    g = AllenIntervalGrammar(event_types=("A", "B"), relations=("<",))
    exp = RuleExpansion(g)
    root = g.root()
    a = exp.expand(root)
    b = exp.expand(root)
    a.N, b.N = 10, 3
    pi = _normalised_visit_distribution(root, temperature=0.0)
    assert pi[a.parent_action] == 1.0
    assert pi.get(b.parent_action, 0.0) == 0.0
