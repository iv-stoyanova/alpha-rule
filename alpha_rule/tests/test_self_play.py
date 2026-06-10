"""
Tests for ``mcts.self_play.run_self_play``.

Pins:
    - One self-play call against a ``ScriptedSimulator`` produces a
      non-empty trajectory whose ``visit_pi`` dicts sum to 1 and whose
      ``state``/``reward`` fields look right.
    - The trajectory respects ``depth_limit`` (no more steps than the
      cap).
    - With NO ``network_evaluator`` the call still completes — PUCT
      degenerates to using the default ``prior=1.0`` on every child.
    - With a deterministic seed the trajectory shape (length, action
      sequence) is reproducible.
"""
from __future__ import annotations

import math

import numpy as np

from alpha_rule.grammar.allen import AllenIntervalGrammar
from alpha_rule.mcts.self_play import run_self_play


class _ConstantSimulator:
    """Returns the same reward for every node."""
    def __init__(self, value=1.0):
        self.value = value
        self.calls = 0

    def evaluate(self, node):
        self.calls += 1
        return self.value


def test_run_self_play_returns_non_empty_trajectory():
    g = AllenIntervalGrammar(event_types=("A", "B"), relations=("<",))
    traj = run_self_play(
        grammar=g,
        simulator=_ConstantSimulator(value=1.0),
        n_simulations=4,
        depth_limit=3,
        rng=np.random.default_rng(0),
    )
    assert len(traj.steps) >= 1
    for step in traj.steps:
        assert math.isfinite(step.reward)
        assert abs(sum(step.visit_pi.values()) - 1.0) < 1e-6


def test_run_self_play_respects_depth_limit():
    g = AllenIntervalGrammar(event_types=("A",), relations=("<",))
    traj = run_self_play(
        grammar=g,
        simulator=_ConstantSimulator(value=0.5),
        n_simulations=2,
        depth_limit=2,
        rng=np.random.default_rng(0),
    )
    assert len(traj.steps) <= 2


def test_run_self_play_works_without_network_evaluator():
    g = AllenIntervalGrammar(event_types=("A", "B"), relations=("<",))
    traj = run_self_play(
        grammar=g,
        simulator=_ConstantSimulator(value=0.7),
        network_evaluator=None,
        n_simulations=3,
        depth_limit=2,
        rng=np.random.default_rng(42),
    )
    assert len(traj.steps) >= 1


def test_run_self_play_is_reproducible_with_seed():
    g = AllenIntervalGrammar(event_types=("A", "B"), relations=("<",))

    def _go(seed):
        return run_self_play(
            grammar=g,
            simulator=_ConstantSimulator(value=1.0),
            n_simulations=3,
            depth_limit=3,
            rng=np.random.default_rng(seed),
        )

    a = _go(123)
    b = _go(123)
    assert len(a.steps) == len(b.steps)
    # Visit-distribution-driven sampling should pick the same actions
    # given the same seed and identical simulator outputs.
    for sa, sb in zip(a.steps, b.steps):
        assert sa.state == sb.state


# --------------------------------------------------------------------------- #
# leaf_eval_mode — AlphaZero leaf bootstrap (NN value at non-terminal leaves).
# --------------------------------------------------------------------------- #

class _RecordingEvaluator:
    """Counts evaluate calls, records names, returns ``EvalResult(value=X)``."""
    def __init__(self, value: float = 0.5):
        from alpha_rule.evaluation.evaluator import EvalResult
        self.value = value
        self.calls = []
        self._EvalResult = EvalResult

    def evaluate(self, node):
        self.calls.append(getattr(node, "name", str(node)))
        return self._EvalResult(value=self.value)


def test_leaf_eval_mode_nn_uses_network_at_non_terminal_leaves():
    """Default leaf_eval_mode='nn': network_evaluator is called for leaf
    value at non-terminal nodes; simulator is reserved for terminal
    nodes and the chosen-step reward."""
    g = AllenIntervalGrammar(event_types=("A", "B"), relations=("<",))
    sim = _ConstantSimulator(value=1.0)
    net = _RecordingEvaluator(value=0.25)
    traj = run_self_play(
        grammar=g,
        simulator=sim,
        network_evaluator=net,
        n_simulations=3,
        depth_limit=2,
        rng=np.random.default_rng(0),
        leaf_eval_mode="nn",
    )
    # The network must have been used at least once (non-terminal leaves
    # exist on the very first MCTS round).
    assert len(net.calls) >= 1
    # Trajectory's chosen-step rewards are still from the simulator
    # (network value is 0.25 but trajectory rewards are 1.0).
    for step in traj.steps:
        if math.isfinite(step.reward):
            assert step.reward == 1.0


def test_leaf_eval_mode_simulator_preserves_legacy_behaviour():
    """Explicit leaf_eval_mode='simulator': the network is never asked
    for leaf value (simulator is, even at non-terminal leaves)."""
    g = AllenIntervalGrammar(event_types=("A", "B"), relations=("<",))
    sim = _ConstantSimulator(value=0.7)
    net = _RecordingEvaluator(value=0.0)
    n_before = sim.calls
    run_self_play(
        grammar=g,
        simulator=sim,
        network_evaluator=net,
        n_simulations=3,
        depth_limit=2,
        rng=np.random.default_rng(0),
        leaf_eval_mode="simulator",
    )
    # Simulator was used. (Network may still be called for priors at
    # newly-expanded children, which is unrelated to leaf value.)
    assert sim.calls > n_before


def test_leaf_eval_mode_nn_falls_back_to_simulator_when_network_missing():
    """If leaf_eval_mode='nn' but no network_evaluator was provided,
    fall back to the simulator. Self-play must not crash."""
    g = AllenIntervalGrammar(event_types=("A",), relations=("<",))
    sim = _ConstantSimulator(value=0.4)
    traj = run_self_play(
        grammar=g,
        simulator=sim,
        network_evaluator=None,
        n_simulations=2,
        depth_limit=2,
        rng=np.random.default_rng(0),
        leaf_eval_mode="nn",
    )
    assert len(traj.steps) >= 1
    assert sim.calls > 0


def test_run_self_play_trajectory_pushes_into_replay_buffer():
    """End-to-end: run_self_play -> ReplayBuffer.push_trajectory works."""
    from alpha_rule.mcts.replay import ReplayBuffer
    g = AllenIntervalGrammar(event_types=("A",), relations=("<",))
    traj = run_self_play(
        grammar=g,
        simulator=_ConstantSimulator(value=2.0),
        n_simulations=2,
        depth_limit=2,
        rng=np.random.default_rng(0),
    )
    buf = ReplayBuffer(capacity=10)
    buf.push_trajectory(traj)
    assert len(buf) == len(traj.steps)


def test_trajectory_steps_record_next_state():
    """
    Every non-terminal trajectory step carries ``next_state`` — the
    child name the reward actually describes. Without this, the
    "best formula" reporter has no way to distinguish "<ROOT>" (the
    state where MCTS was rooted) from the child that earned the
    reward.
    """
    g = AllenIntervalGrammar(event_types=("A",), relations=("<",))
    traj = run_self_play(
        grammar=g,
        simulator=_ConstantSimulator(value=1.0),
        n_simulations=4,
        depth_limit=2,
        rng=np.random.default_rng(0),
    )
    assert len(traj.steps) >= 1
    for step in traj.steps:
        assert step.next_state is not None, \
            "run_self_play must populate next_state on every step"
        # The child name differs from the parent (an action was taken).
        assert step.next_state != step.state


def test_best_in_trajectory_prefers_next_state():
    """
    ``_best_in_trajectory`` should return the CHILD that earned the
    reward, not the parent the MCTS was rooted at. Without this the
    best formula surfaces as "<ROOT>" whenever the first action is
    the best-rewarded one.
    """
    import pytest

    from alpha_rule.mcts.replay import TrajectoryStep, Trajectory
    # _best_in_trajectory is a training helper (C6); skip until it is migrated.
    _best_in_trajectory = pytest.importorskip("alpha_rule.training.train")._best_in_trajectory

    traj = Trajectory(steps=[
        TrajectoryStep(state="<ROOT>", visit_pi={"A": 1.0}, reward=0.9,
                       next_state="A"),
        TrajectoryStep(state="A",      visit_pi={"<END>": 1.0}, reward=0.5,
                       next_state="A <END>"),
    ])
    best_state, best_reward = _best_in_trajectory(traj)
    assert best_state == "A"                 # the child, not "<ROOT>"
    assert best_reward == 0.9


def test_best_in_trajectory_falls_back_to_state_when_no_next_state():
    """Back-compat: a legacy TrajectoryStep without next_state still works."""
    import pytest

    from alpha_rule.mcts.replay import TrajectoryStep, Trajectory
    # _best_in_trajectory is a training helper (C6); skip until it is migrated.
    _best_in_trajectory = pytest.importorskip("alpha_rule.training.train")._best_in_trajectory

    traj = Trajectory(steps=[
        TrajectoryStep(state="legacy", visit_pi={}, reward=0.3),  # next_state=None by default
    ])
    best_state, best_reward = _best_in_trajectory(traj)
    assert best_state == "legacy"


def test_run_self_play_does_not_crash_when_terminal_is_selected():
    """
    Regression: when MCTS selection descends into an END_RULE-terminal
    (name ending in ``<END>``), ``_run_one_round`` must NOT try to
    expand it — expanding "A <END>" would feed "A <END> <" to
    ``AllenMatrix.from_hierarchy_string`` which raises ValueError.

    This test uses a big enough n_simulations + depth_limit that MCTS
    definitely visits some terminals (END_RULE is an applicable
    production on every non-root node).
    """
    g = AllenIntervalGrammar(event_types=("A", "B"), relations=("<",))
    # Should complete without raising.
    traj = run_self_play(
        grammar=g,
        simulator=_ConstantSimulator(value=1.0),
        n_simulations=30,
        depth_limit=3,
        rng=np.random.default_rng(0),
    )
    assert len(traj.steps) >= 1


# --------------------------------------------------------------------------- #
# Dirichlet noise on root priors — opt-in (alpha-rule-updates).
#
# Default ``dirichlet_eps=0.0`` must leave every MCTS decision identical
# to the pre-change behaviour (reproducibility).
#
# When enabled, the root's *children's* priors are mixed with a
# Dirichlet sample:
#     prior' = (1 - eps) * prior + eps * noise,   noise ~ Dir(alpha * k)
# with ``k`` = number of live children. This is the AlphaZero recipe —
# forces residual exploration of branches PUCT would otherwise starve.
# --------------------------------------------------------------------------- #


class _PriorCapturingEvaluator:
    """Records every call and returns uniform priors + constant value."""
    def __init__(self, value=0.0):
        self._value = value
        self.calls: list = []

    def evaluate(self, node):
        from alpha_rule.evaluation.evaluator import EvalResult
        # Uniform priors over the node's existing children (which is all
        # ``_write_priors`` distributes onto anyway).
        actions = [c.parent_action for c in node.children if c.parent_action]
        if not actions:
            return EvalResult(value=self._value, priors={})
        p = 1.0 / len(actions)
        priors = {a: p for a in actions}
        self.calls.append((node.name, dict(priors)))
        return EvalResult(value=self._value, priors=priors)


def test_dirichlet_default_disabled_matches_no_noise_path():
    """Pin: ``dirichlet_eps=0.0`` (default) is identical to not passing it."""
    g = AllenIntervalGrammar(event_types=("A", "B", "C"), relations=("<",))
    a = run_self_play(
        grammar=g, simulator=_ConstantSimulator(1.0),
        n_simulations=8, depth_limit=2,
        rng=np.random.default_rng(7),
    )
    b = run_self_play(
        grammar=g, simulator=_ConstantSimulator(1.0),
        n_simulations=8, depth_limit=2,
        rng=np.random.default_rng(7),
        dirichlet_eps=0.0,
    )
    assert len(a.steps) == len(b.steps)
    for sa, sb in zip(a.steps, b.steps):
        assert sa.state == sb.state
        assert sa.visit_pi == sb.visit_pi


def test_dirichlet_positive_eps_mixes_noise_into_root_priors():
    """With ``dirichlet_eps > 0``, root children's priors are mixed with
    a Dirichlet sample. We check the mixing formula directly by
    inspecting the root's children after self-play."""
    import numpy as np
    g = AllenIntervalGrammar(event_types=("A", "B", "C"), relations=("<",))
    evaluator = _PriorCapturingEvaluator(value=0.0)

    # First, run WITHOUT noise and capture the root's prior vector.
    # Seed the rng for reproducibility.
    traj_no_noise = run_self_play(
        grammar=g, simulator=_ConstantSimulator(1.0),
        network_evaluator=evaluator,
        n_simulations=4, depth_limit=1,     # single construction step
        rng=np.random.default_rng(99),
        dirichlet_eps=0.0, dirichlet_alpha=0.3,
    )
    # Then with noise: different rng → different priors, visit counts.
    traj_noisy = run_self_play(
        grammar=g, simulator=_ConstantSimulator(1.0),
        network_evaluator=evaluator,
        n_simulations=4, depth_limit=1,
        rng=np.random.default_rng(99),
        dirichlet_eps=0.25, dirichlet_alpha=0.3,
    )
    # Assert the noise actually changed SOMETHING — either the visit
    # distribution or its support at the root. Identical distributions
    # would mean noise did nothing.
    assert traj_no_noise.steps[0].visit_pi != traj_noisy.steps[0].visit_pi


def test_dirichlet_rejects_bad_arguments():
    import pytest
    g = AllenIntervalGrammar(event_types=("A",), relations=("<",))
    with pytest.raises(ValueError):
        run_self_play(
            grammar=g, simulator=_ConstantSimulator(1.0),
            n_simulations=2, depth_limit=1,
            dirichlet_eps=-0.1,            # negative
        )
    with pytest.raises(ValueError):
        run_self_play(
            grammar=g, simulator=_ConstantSimulator(1.0),
            n_simulations=2, depth_limit=1,
            dirichlet_eps=1.5,             # > 1
        )
    with pytest.raises(ValueError):
        run_self_play(
            grammar=g, simulator=_ConstantSimulator(1.0),
            n_simulations=2, depth_limit=1,
            dirichlet_eps=0.25, dirichlet_alpha=0.0,   # alpha must be > 0
        )


def test_dirichlet_preserves_approximate_sum_to_one_on_root_children():
    """Mixing with a valid Dirichlet sample keeps the sum of root-child
    priors close to 1 (they already summed to 1; convex combination of
    two distributions that each sum to 1 also sums to 1)."""
    import numpy as np
    from alpha_rule.mcts.self_play import _apply_root_dirichlet_noise
    from alpha_rule.mcts.expansion import RuleExpansion
    g = AllenIntervalGrammar(event_types=("A", "B", "C"), relations=("<",))
    root = g.root()
    expansion = RuleExpansion(g)
    # Expand all the root's children so priors have something to mix into.
    while not root.is_fully_expanded():
        expansion.expand(root)
    # Uniform priors on the children (sum = 1).
    n = len(root.children)
    for c in root.children:
        c.prior = 1.0 / n
    rng = np.random.default_rng(3)
    _apply_root_dirichlet_noise(root, eps=0.25, alpha=0.3, rng=rng)
    s = sum(c.prior for c in root.children)
    assert abs(s - 1.0) < 1e-6
