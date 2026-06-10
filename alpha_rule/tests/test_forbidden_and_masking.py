"""
Tests for the search-masking knobs in ``run_self_play``:

    forbidden_root_actions  -- root-only, matched by parent_action (token)
    dead_rule_names         -- anywhere, matched by full node.name

Both mark the matching child ``is_dead`` so PUCT never visits it, it gets no
simulator call, and it is absent from the policy target ``visit_pi``.
"""
from __future__ import annotations

import numpy as np

from alpha_rule.grammar.allen import AllenIntervalGrammar
from alpha_rule.mcts.backprop import MaxRewardBackup
from alpha_rule.mcts.expansion import RuleExpansion
from alpha_rule.mcts.selection import PUCTSelection
from alpha_rule.mcts.self_play import _run_one_round, run_self_play


class _ConstantSimulator:
    def __init__(self, value=1.0):
        self.value = value

    def evaluate(self, node):
        return self.value


class _RecordingSimulator(_ConstantSimulator):
    """Records the name of every node it is asked to evaluate."""

    def __init__(self, value=1.0):
        super().__init__(value)
        self.names = []

    def evaluate(self, node):
        self.names.append(node.name)
        return super().evaluate(node)


# --------------------------------------------------------------------------- #
# forbidden_root_actions
# --------------------------------------------------------------------------- #

def test_forbidden_root_action_never_chosen_or_in_pi():
    g = AllenIntervalGrammar(event_types=("A", "B", "C"), relations=("<",))
    traj = run_self_play(
        grammar=g, simulator=_ConstantSimulator(1.0),
        n_simulations=8, depth_limit=1, rng=np.random.default_rng(0),
        forbidden_root_actions=["A"], leaf_eval_mode="simulator",
    )
    s0 = traj.steps[0]
    assert s0.state == "<ROOT>"
    assert "A" not in s0.visit_pi
    assert s0.next_state != "A"
    assert set(s0.visit_pi) <= {"B", "C"}
    assert abs(sum(s0.visit_pi.values()) - 1.0) < 1e-6


def test_forbidden_root_child_is_dead_and_unvisited():
    g = AllenIntervalGrammar(event_types=("A", "B"), relations=("<",))
    exp = RuleExpansion(g)
    root = g.root()
    while not root.is_fully_expanded():
        exp.expand(root)
    forbidden = next(c for c in root.children if c.parent_action == "A")
    forbidden.is_dead = True
    _run_one_round(
        root, n_simulations=20, simulator=_ConstantSimulator(1.0),
        network_evaluator=None, selection=PUCTSelection(),
        backup=MaxRewardBackup(), expansion=exp, leaf_eval_mode="simulator",
    )
    assert forbidden.is_dead is True
    assert forbidden.N == 0              # PUCT never descended into it
    assert forbidden.children == []      # never expanded under it
    live = next(c for c in root.children if c.parent_action == "B")
    assert live.N > 0


def test_all_but_one_root_action_forbidden_forces_choice():
    g = AllenIntervalGrammar(event_types=("A", "B", "C"), relations=("<",))
    traj = run_self_play(
        grammar=g, simulator=_ConstantSimulator(1.0),
        n_simulations=8, depth_limit=1, rng=np.random.default_rng(0),
        forbidden_root_actions=["A", "B"], leaf_eval_mode="simulator",
    )
    s0 = traj.steps[0]
    assert list(s0.visit_pi) == ["C"]
    assert abs(s0.visit_pi["C"] - 1.0) < 1e-9
    assert s0.next_state == "C"


def test_forbidden_root_action_is_root_scoped():
    """Forbidding the token "A" at the root must NOT forbid it deeper (the node
    "B A" further down is a different rule and stays live)."""
    g = AllenIntervalGrammar(event_types=("A", "B"), relations=("<",))
    exp = RuleExpansion(g)
    root = g.root()
    while not root.is_fully_expanded():
        exp.expand(root)
    root_a = next(c for c in root.children if c.parent_action == "A")
    root_a.is_dead = True                          # the forbidden marking
    b = next(c for c in root.children if c.parent_action == "B")
    while not b.is_fully_expanded():
        exp.expand(b)
    deep_a = next(c for c in b.children if c.parent_action == "A")
    assert root_a.is_dead is True
    assert deep_a.is_dead is False
    assert deep_a.name == "B A"


def test_forbidding_all_root_actions_is_empty_with_no_sim_calls():
    """Edge case: forbidding every root action ends the episode immediately and
    -- thanks to the dead-root guard -- spends no simulator calls re-evaluating
    the root."""
    g = AllenIntervalGrammar(event_types=("A", "B"), relations=("<",))
    sim = _RecordingSimulator(1.0)
    traj = run_self_play(
        grammar=g, simulator=sim, n_simulations=7, depth_limit=2,
        rng=np.random.default_rng(0), forbidden_root_actions=["A", "B"],
        leaf_eval_mode="simulator",
    )
    assert traj.steps == []
    assert sim.names == []


# --------------------------------------------------------------------------- #
# dead_rule_names
# --------------------------------------------------------------------------- #

def test_dead_rule_names_skips_simulator_and_pi():
    g = AllenIntervalGrammar(event_types=("A", "B"), relations=("<",))
    sim = _RecordingSimulator(1.0)
    traj = run_self_play(
        grammar=g, simulator=sim, n_simulations=15, depth_limit=1,
        rng=np.random.default_rng(0), dead_rule_names={"A"},
        leaf_eval_mode="simulator",
    )
    assert "A" not in sim.names           # no simulator call on the known-dead rule
    s0 = traj.steps[0]
    assert "A" not in s0.visit_pi
    assert s0.next_state != "A"


def test_dead_rule_names_matches_full_name_not_token():
    """``dead_rule_names`` matches the whole rule name, not an action token: the
    root rule "A" is masked, but the longer rule "A A" is a different name."""
    g = AllenIntervalGrammar(event_types=("A",), relations=("<",))
    sim = _RecordingSimulator(1.0)
    run_self_play(
        grammar=g, simulator=sim, n_simulations=20, depth_limit=2,
        rng=np.random.default_rng(0), dead_rule_names={"A A"},
        leaf_eval_mode="simulator",
    )
    assert "A" in sim.names               # root rule "A" is NOT masked (name != "A A")
    assert "A A" not in sim.names         # the rule literally named "A A" IS masked


def test_forbidden_plus_dead_rule_names_interaction():
    g = AllenIntervalGrammar(event_types=("A", "B", "C"), relations=("<",))
    traj = run_self_play(
        grammar=g, simulator=_ConstantSimulator(1.0),
        n_simulations=8, depth_limit=1, rng=np.random.default_rng(0),
        forbidden_root_actions=["A"], dead_rule_names={"B"},
        leaf_eval_mode="simulator",
    )
    s0 = traj.steps[0]
    assert list(s0.visit_pi) == ["C"]     # A forbidden, B dead-named -> only C live
    assert s0.next_state == "C"
