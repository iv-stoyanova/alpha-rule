"""
Tests for the fixes addressing the length-1-best-rule bias:

1. ``run_self_play(n_chosen_evals=N)`` — averages N independent
   simulator samples for the chosen-step reward to beat Q-learning
   variance. ``_multi_sample_chosen_reward`` simply averages the N finite
   samples of ``simulator.evaluate(node)``; the tests drive it with a
   *stateful* stub evaluator that returns different values across calls.

2. ``_best_in_trajectory`` — on **ties** in finite reward the **longer
   rule** (more tokens, per ``name.split()``) wins, countering the
   tendency of length-1 rules to hit the reward ceiling first and
   then block more-specific ones that match equally well.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from alpha_rule.grammar.allen import AllenIntervalGrammar
from alpha_rule.mcts.replay import Trajectory, TrajectoryStep
from alpha_rule.mcts.self_play import run_self_play
from alpha_rule.training.train import _best_in_trajectory


# --------------------------------------------------------------------------- #
# 1) _best_in_trajectory — tie-break prefers longer rule
# --------------------------------------------------------------------------- #

def test_best_in_trajectory_ties_prefer_longer_rule():
    """On equal finite reward, the longer (more-token) rule wins."""
    traj = Trajectory(steps=[
        TrajectoryStep(state="<ROOT>", visit_pi={}, reward=0.5, next_state="A"),
        TrajectoryStep(state="A",      visit_pi={}, reward=0.5, next_state="A < B"),
    ])
    best_state, best_reward = _best_in_trajectory(traj)
    assert best_state == "A < B"
    assert best_reward == 0.5


def test_best_in_trajectory_strict_reward_still_beats_length():
    """A higher reward wins even if the rule is shorter."""
    traj = Trajectory(steps=[
        TrajectoryStep(state="<ROOT>", visit_pi={}, reward=0.7, next_state="A"),
        TrajectoryStep(state="A",      visit_pi={}, reward=0.5, next_state="A < B"),
    ])
    best_state, best_reward = _best_in_trajectory(traj)
    assert best_state == "A"
    assert best_reward == 0.7


def test_best_in_trajectory_three_way_tie_takes_longest():
    traj = Trajectory(steps=[
        TrajectoryStep(state="<ROOT>", visit_pi={}, reward=-0.3, next_state="A"),
        TrajectoryStep(state="A",      visit_pi={}, reward=-0.3, next_state="A <"),
        TrajectoryStep(state="A <",    visit_pi={}, reward=-0.3, next_state="A < B"),
    ])
    best_state, best_reward = _best_in_trajectory(traj)
    assert best_state == "A < B"
    assert best_reward == -0.3


def test_cross_iteration_running_best_ties_prefer_longer_rule():
    """
    Across iterations, ``train()`` must also replace a length-1
    running-best with an equally-scored longer rule found later.
    Without this, iteration 0's ``"A"`` locks in forever even if
    iteration 1 finds ``"A < B"`` with the same reward.
    """
    from alpha_rule.training.train import train

    # Return a value depending on the rule's token count so iter 0's
    # single-token rule and iter 1's longer rule both hit the same ceiling.
    class _FixedValSim:
        def evaluate(self, node):
            # Same reward for every rule — tests tie-break behaviour.
            return 0.0

    g = AllenIntervalGrammar(event_types=("A", "B"), relations=("<",))

    log = train(
        grammar=g,
        expensive_simulator=_FixedValSim(),
        n_iterations=4,
        n_simulations=4,
        depth_limit=3,        # room for 3-token rules
        seed=0,
        buffer_warmup=1,
        train_steps_per_iteration=1,
        max_len=12,
        d_model=16,
        nhead=2,
        num_layers=1,
    )

    # Every evaluation returns 0.0 so `best_reward` is 0.0 throughout.
    assert log.best_reward == 0.0
    assert log.best_formula is not None
    # Across 4 iterations with depth_limit=3, the tie-break should have
    # promoted a multi-token rule over whatever length-1 rule iter 0
    # produced first.
    best_len = len(log.best_formula.split())
    assert best_len >= 2, (
        f"expected a multi-token running-best rule after 4 iterations with "
        f"tie-break, got {log.best_formula!r} (len={best_len})"
    )


def test_best_in_trajectory_endrule_counts_as_token():
    """
    ``"A <END>"`` has 2 tokens via split(); ``"A"`` has 1. The
    explicit terminator counts so, at equal reward, the terminated
    rule is preferred over a bare single-event rule. This is the
    whole point — length-1 rules lose ties.
    """
    traj = Trajectory(steps=[
        TrajectoryStep(state="<ROOT>", visit_pi={}, reward=-0.1, next_state="A"),
        TrajectoryStep(state="A",      visit_pi={}, reward=-0.1, next_state="A <END>"),
    ])
    best_state, _ = _best_in_trajectory(traj)
    assert best_state == "A <END>"


# --------------------------------------------------------------------------- #
# 2) run_self_play — n_chosen_evals averages multiple samples
# --------------------------------------------------------------------------- #

class _SeqSim:
    """Evaluator that returns a deterministic sequence; counts calls."""
    def __init__(self, values):
        self.values = list(values)
        self.calls = 0
    def evaluate(self, node):
        v = self.values[self.calls % len(self.values)]
        self.calls += 1
        return v


def test_default_n_chosen_evals_is_one():
    """Existing behaviour: exactly one chosen-step sim.evaluate() call."""
    sim = _SeqSim([1.0])
    grammar = AllenIntervalGrammar(event_types=("A",), relations=("<",))
    traj = run_self_play(
        grammar=grammar, simulator=sim,
        n_simulations=1, depth_limit=1,
        rng=np.random.default_rng(0),
    )
    # 1 MCTS sim + 1 chosen-step eval = 2 calls total.
    assert len(traj.steps) == 1
    assert sim.calls == 2
    assert traj.steps[0].reward == 1.0


def test_n_chosen_evals_multiplies_chosen_step_calls():
    """n_chosen_evals=3 adds 2 extra calls on top of the single default."""
    grammar = AllenIntervalGrammar(event_types=("A",), relations=("<",))

    sim_baseline = _SeqSim([1.0])
    run_self_play(
        grammar=grammar, simulator=sim_baseline,
        n_simulations=1, depth_limit=1,
        rng=np.random.default_rng(0),
    )

    sim_multi = _SeqSim([1.0])
    run_self_play(
        grammar=grammar, simulator=sim_multi,
        n_simulations=1, depth_limit=1,
        n_chosen_evals=3,
        rng=np.random.default_rng(0),
    )
    assert sim_multi.calls == sim_baseline.calls + 2


def test_n_chosen_evals_averages_finite_samples():
    """
    With a sequence [10, 20, 30] over 3 chosen-step evals, the recorded
    reward is the mean (20.0).
    ``n_simulations=0`` skips MCTS rollouts so only the chosen step
    calls the simulator. The sim still has to cover the first visit-dist
    read; the implementation is expected to tolerate that (see
    ``run_self_play``'s own handling of empty visit_pi).
    """
    class _CollectSim:
        """Returns next value; never errors."""
        def __init__(self, vals):
            self.vals = list(vals)
            self.i = 0
            self.calls = 0
        def evaluate(self, node):
            v = self.vals[self.i % len(self.vals)]
            self.i += 1
            self.calls += 1
            return v

    grammar = AllenIntervalGrammar(event_types=("A",), relations=("<",))
    sim = _CollectSim([10.0, 20.0, 30.0])

    traj = run_self_play(
        grammar=grammar, simulator=sim,
        n_simulations=1, depth_limit=1,
        n_chosen_evals=3,
        rng=np.random.default_rng(0),
    )
    # One MCTS call consumed value 10; then 3 chosen-step calls
    # consumed 20, 30, 10 (wrap) → mean = 20.0.
    assert len(traj.steps) == 1
    assert traj.steps[0].reward == pytest.approx(20.0)


def test_n_chosen_evals_ignores_minus_inf_when_averaging():
    """
    If the evaluator sometimes returns -inf, the average should use
    only the finite samples. MCTS needs a finite sample first so the
    tree stays alive; then the chosen-step calls are the ones we
    inspect.
    """
    class _MixedSim:
        def __init__(self, vals):
            self.vals = list(vals)
            self.i = 0
        def evaluate(self, node):
            v = self.vals[self.i % len(self.vals)]
            self.i += 1
            return v

    grammar = AllenIntervalGrammar(event_types=("A",), relations=("<",))
    # Call 0 = MCTS rollout (finite 0.0, keeps tree alive).
    # Calls 1,2,3 = chosen-step evals: -inf, 0.4, -inf → finite mean = 0.4.
    sim = _MixedSim([0.0, -math.inf, 0.4, -math.inf])
    traj = run_self_play(
        grammar=grammar, simulator=sim,
        n_simulations=1, depth_limit=1,
        n_chosen_evals=3,
        rng=np.random.default_rng(0),
    )
    assert len(traj.steps) == 1
    assert traj.steps[0].reward == pytest.approx(0.4)


def test_n_chosen_evals_all_minus_inf_stays_minus_inf():
    """
    If every chosen-step sample is -inf, the recorded reward must
    stay -inf (no zero-count average crash, no spurious 0.0).
    """
    class _OnePhaseSim:
        def __init__(self):
            self.calls = 0
        def evaluate(self, node):
            self.calls += 1
            # First call (MCTS) is finite so the tree stays alive;
            # subsequent chosen-step samples are all -inf.
            return 0.0 if self.calls == 1 else -math.inf

    grammar = AllenIntervalGrammar(event_types=("A",), relations=("<",))
    traj = run_self_play(
        grammar=grammar, simulator=_OnePhaseSim(),
        n_simulations=1, depth_limit=1,
        n_chosen_evals=5,
        rng=np.random.default_rng(0),
    )
    assert len(traj.steps) == 1
    assert traj.steps[0].reward == -math.inf
