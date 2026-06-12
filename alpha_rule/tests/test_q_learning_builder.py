"""
Tests for ``alpha_rule.policy_agents.q_learning.builder.q_learning_agent_builder``.

Pins:
    - Default-disabled early-stop preserves the loop count exactly.
    - With ``early_stop_tol > 0`` the loop breaks once the Q-table is
      stable for ``patience`` consecutive checks.
    - ``get_state_tuple`` accepts a pre-resolved ``get_otc`` (so the
      builder can cache the wrapper-chain walk).
"""
from __future__ import annotations

import numpy as np

try:
    import gymnasium as gym
    _HAS_GYM = True
except ImportError:
    _HAS_GYM = False


class _Discrete:
    """Minimal stand-in for gym.spaces.Discrete that survives isinstance."""

    def __init__(self, n):
        self.n = n
        self._rng = np.random.default_rng(0)

    def sample(self):
        return int(self._rng.integers(self.n))


class _MiniOTC:
    """Stand-in for the OTC env exposing ``get_observations``."""

    def __init__(self, open_state=(0, 0)):
        self._open = list(open_state)

    def get_observations(self):
        return {"open": tuple(self._open)}


class _StepCountingEnv:
    """
    Minimal env that q_learning_agent_builder can drive.

    - ``action_space``: ``Discrete(2)`` (we monkey-patch isinstance below).
    - ``reset`` returns a constant context obs of length 1.
    - ``step`` returns a constant reward (configurable; default 0.0).
    - ``get_otc()`` returns a constant ``_MiniOTC``.

    Counts ``step`` invocations on ``self.steps`` so tests can assert on
    the loop length.
    """

    def __init__(self, reward=0.0, episode_len=10):
        self.action_space = _Discrete(2)
        self._ctx = np.array([0], dtype=np.int8)
        self._otc = _MiniOTC()
        self.steps = 0
        self.resets = 0
        self._t = 0
        self._reward = reward
        self._episode_len = episode_len

    def reset(self):
        self.resets += 1
        self._t = 0
        return self._ctx

    def step(self, action):
        self.steps += 1
        self._t += 1
        done = self._t >= self._episode_len
        return self._ctx, self._reward, done, False, {}

    def get_otc(self):
        return self._otc


def _patch_discrete_isinstance():
    """The builder asserts ``isinstance(env.action_space, gym.spaces.Discrete)``.
    Make our mini Discrete pass that check by aliasing the gym class.

    Returns the original class so callers can restore it.
    """
    if not _HAS_GYM:
        return None
    orig = gym.spaces.Discrete
    gym.spaces.Discrete = _Discrete
    return orig


def _restore_discrete(orig):
    if orig is not None:
        gym.spaces.Discrete = orig


def test_q_learning_default_runs_full_timesteps():
    """Default ``early_stop_tol=0.0`` keeps the loop at exactly ``total_timesteps`` steps."""
    if not _HAS_GYM:
        return
    orig = _patch_discrete_isinstance()
    try:
        from alpha_rule.policy_agents.q_learning.builder import q_learning_agent_builder

        env = _StepCountingEnv(reward=0.0, episode_len=100)
        q_table, policy = q_learning_agent_builder(env, total_timesteps=300)
        assert env.steps == 300, f"expected 300 steps, got {env.steps}"
        # Policy is callable and returns an int action.
        first_state = next(iter(q_table.keys()))
        assert isinstance(policy(first_state), int)
    finally:
        _restore_discrete(orig)


def test_q_learning_early_stop_breaks_loop():
    """With zero reward the Q-table never moves; early-stop should fire
    after ``patience`` consecutive sub-tolerance checks."""
    if not _HAS_GYM:
        return
    orig = _patch_discrete_isinstance()
    try:
        from alpha_rule.policy_agents.q_learning.builder import q_learning_agent_builder

        env = _StepCountingEnv(reward=0.0, episode_len=100)
        q_table, _policy = q_learning_agent_builder(
            env,
            total_timesteps=10_000,    # big budget; should NOT be exhausted
            early_stop_tol=1e-6,
            check_interval=50,
            patience=2,
        )
        # 2 patience × 50 interval = ~100 steps once the table is stable;
        # in practice the first check is a no-op (no prev snapshot) so
        # ~150 steps. Either way, < 10_000.
        assert env.steps < 10_000, (
            f"expected early stop well under 10k steps, got {env.steps}"
        )
        assert env.steps >= 50, "should run at least one check_interval"
    finally:
        _restore_discrete(orig)


def test_q_learning_respects_total_timesteps_cap_even_with_changing_q():
    """With ``early_stop_tol`` set but a Q-table that keeps changing
    (non-zero reward), the loop runs to ``total_timesteps``."""
    if not _HAS_GYM:
        return
    orig = _patch_discrete_isinstance()
    try:
        from alpha_rule.policy_agents.q_learning.builder import q_learning_agent_builder

        env = _StepCountingEnv(reward=1.0, episode_len=10)
        q_table, _policy = q_learning_agent_builder(
            env,
            total_timesteps=400,
            early_stop_tol=1e-6,
            check_interval=100,
            patience=3,
        )
        # Reward of 1.0 every step means Q-values keep climbing; early stop
        # should NOT trigger within 400 steps.
        assert env.steps == 400, (
            f"expected full 400 steps under non-converging Q; got {env.steps}"
        )
    finally:
        _restore_discrete(orig)


def test_get_state_tuple_accepts_cached_get_otc():
    """``get_state_tuple`` should produce the same output whether
    ``get_otc`` is passed explicitly or resolved from the env chain."""
    if not _HAS_GYM:
        return
    from alpha_rule.policy_agents.q_learning.builder import get_state_tuple

    env = _StepCountingEnv()
    explicit = get_state_tuple(env, env._ctx, get_otc=env.get_otc)
    resolved = get_state_tuple(env, env._ctx)
    assert explicit == resolved
