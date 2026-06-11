"""
Tests for persistent dead-rule masking across self-play episodes.

Pins:
    - ``run_self_play`` accepts ``dead_rule_names`` and never asks the
      simulator for a rule it already knows is dead.
    - Pre-expanded root children (Dirichlet path) honour the same set.
    - ``train()`` accumulates -inf rule names across iterations.
"""
from __future__ import annotations

import math

import numpy as np

from alpha_rule.grammar.allen import AllenIntervalGrammar
from alpha_rule.mcts.self_play import run_self_play
from alpha_rule.training.train import _collect_dead_rules
from alpha_rule.mcts.replay import Trajectory, TrajectoryStep


class _FailFastSimulator:
    """Simulator that returns -inf for any rule name in ``failing_names``,
    a positive value otherwise. Records every name it was asked about."""

    def __init__(self, failing_names, positive_value=1.0):
        self.failing_names = set(failing_names)
        self.positive_value = positive_value
        self.calls = []

    def evaluate(self, node):
        name = node.name
        self.calls.append(name)
        if name in self.failing_names:
            return float("-inf")
        return self.positive_value


def test_dead_rule_names_marks_expanded_children_dead():
    """A child whose name is in ``dead_rule_names`` is marked
    ``is_dead=True`` on expansion and the simulator is never asked
    about it."""
    g = AllenIntervalGrammar(event_types=("A", "B"), relations=("<",))

    # Baseline: without dead set, "A" gets evaluated.
    sim_base = _FailFastSimulator(failing_names={"A"})
    run_self_play(
        grammar=g,
        simulator=sim_base,
        n_simulations=6,
        depth_limit=2,
        rng=np.random.default_rng(0),
    )
    assert "A" in sim_base.calls, (
        "test premise broken: baseline should have called simulator on 'A'"
    )

    # Now with dead set pre-populated — same seed, identical RNG path.
    sim_dead = _FailFastSimulator(failing_names={"A"})
    run_self_play(
        grammar=g,
        simulator=sim_dead,
        n_simulations=6,
        depth_limit=2,
        rng=np.random.default_rng(0),
        dead_rule_names={"A"},
    )
    assert "A" not in sim_dead.calls, (
        f"simulator was asked about a known-dead rule: calls={sim_dead.calls}"
    )


def test_dead_rule_names_none_preserves_default_behaviour():
    """Passing ``dead_rule_names=None`` (default) leaves behaviour
    bit-identical to not passing the kwarg at all."""
    g = AllenIntervalGrammar(event_types=("A", "B"), relations=("<",))

    def _go(seed):
        sim = _FailFastSimulator(failing_names=set())
        return run_self_play(
            grammar=g,
            simulator=sim,
            n_simulations=4,
            depth_limit=2,
            rng=np.random.default_rng(seed),
        )

    def _go_with_none(seed):
        sim = _FailFastSimulator(failing_names=set())
        return run_self_play(
            grammar=g,
            simulator=sim,
            n_simulations=4,
            depth_limit=2,
            rng=np.random.default_rng(seed),
            dead_rule_names=None,
        )

    a = _go(42)
    b = _go_with_none(42)
    assert len(a.steps) == len(b.steps)
    for sa, sb in zip(a.steps, b.steps):
        assert sa.state == sb.state
        assert sa.visit_pi == sb.visit_pi


def test_collect_dead_rules_extracts_inf_step_names():
    """``_collect_dead_rules`` returns rule names from steps whose
    reward was non-finite."""
    traj = Trajectory(steps=[
        TrajectoryStep(state="<ROOT>", visit_pi={"A": 1.0}, reward=0.5,
                       next_state="A"),
        TrajectoryStep(state="A", visit_pi={"<": 1.0}, reward=float("-inf"),
                       next_state="A <"),
        TrajectoryStep(state="A <", visit_pi={"B": 1.0}, reward=0.7,
                       next_state="A < B"),
    ])
    dead = _collect_dead_rules(traj)
    assert dead == ["A <"]


def test_collect_dead_rules_handles_node_like_objects():
    """``next_state`` may be a node with ``.name`` rather than a raw string."""
    class _N:
        def __init__(self, name):
            self.name = name

    traj = Trajectory(steps=[
        TrajectoryStep(state=_N("<ROOT>"), visit_pi={"A": 1.0},
                       reward=float("-inf"), next_state=_N("A")),
    ])
    dead = _collect_dead_rules(traj)
    assert dead == ["A"]


def test_collect_dead_rules_ignores_finite_rewards():
    """Steps with finite rewards do not contribute to the dead set."""
    traj = Trajectory(steps=[
        TrajectoryStep(state="X", visit_pi={"a": 1.0}, reward=0.0,
                       next_state="Y"),
        TrajectoryStep(state="Y", visit_pi={"a": 1.0}, reward=1.5,
                       next_state="Z"),
    ])
    assert _collect_dead_rules(traj) == []
