"""
Tests for the Q-learning agent-evaluation functions.

Pins:
    - ``eval_total_reward`` returns the mean episodic reward (no +50 bonus)
      when the candidate rule fires, and ``-inf`` when it never fires.
    - ``eval_mean_reward_success_steps`` returns its 3-tuple when the candidate
      fires, and ``-inf`` (a scalar) when it never fires.
    - The candidate rule is the LAST column of the context observation.
"""
from __future__ import annotations

import math

import numpy as np


class _MiniOTC:
    """Stand-in OTC exposing ``get_observations`` (box-open bits)."""
    def get_observations(self):
        return {"open": (0, 0)}


class _RuleEvalEnv:
    """Minimal env whose context observation is a fixed rule-indicator vector.

    ``indicator[-1]`` is the candidate rule; a value of 1 means it fires.
    """
    def __init__(self, indicator, reward=1.0, episode_len=2):
        self._ind = np.array(indicator, dtype=np.int8)
        self._reward = reward
        self._len = episode_len
        self._otc = _MiniOTC()
        self._t = 0

    def reset(self, **kwargs):
        self._t = 0
        return self._ind, {}

    def step(self, action):
        self._t += 1
        return self._ind, self._reward, self._t >= self._len, False, {}

    def get_otc(self):
        return self._otc


def _agent():
    return (None, lambda full_state: 0)        # (q_table, policy); policy ignored


# --------------------------------------------------------------------------- #
# eval_total_reward: no +50 bonus; -inf when the candidate never fires
# --------------------------------------------------------------------------- #

def test_eval_total_reward_returns_plain_mean_when_candidate_fires():
    from alpha_rule.policy_agents.q_learning.eval_total_reward import (
        q_learning_agent_eval_total_reward,
    )
    env = _RuleEvalEnv(indicator=(1,), reward=1.0, episode_len=2)
    score = q_learning_agent_eval_total_reward(_agent(), env, n_eval_episodes=3)
    # 2 steps * reward 1.0 = 2.0 per episode; mean = 2.0, NO +50 bonus.
    assert score == 2.0


def test_eval_total_reward_minus_inf_when_candidate_never_fires():
    from alpha_rule.policy_agents.q_learning.eval_total_reward import (
        q_learning_agent_eval_total_reward,
    )
    env = _RuleEvalEnv(indicator=(0,), reward=1.0, episode_len=2)
    score = q_learning_agent_eval_total_reward(_agent(), env, n_eval_episodes=3)
    assert math.isinf(score) and score < 0


# --------------------------------------------------------------------------- #
# eval_mean_reward_success_steps: -inf when the candidate never fires
# --------------------------------------------------------------------------- #

def test_eval_mean_success_returns_tuple_when_candidate_fires():
    from alpha_rule.policy_agents.q_learning.eval_mean_success import (
        q_learning_agent_eval_mean_reward_success_steps,
    )
    env = _RuleEvalEnv(indicator=(1,), reward=1.0, episode_len=2)
    out = q_learning_agent_eval_mean_reward_success_steps(_agent(), env, n_eval_episodes=3)
    assert isinstance(out, tuple) and len(out) == 3
    mean_reward, success_rate, mean_steps = out
    assert mean_reward == 2.0 and success_rate == 1.0 and mean_steps == 2.0


def test_eval_mean_success_minus_inf_when_candidate_never_fires():
    from alpha_rule.policy_agents.q_learning.eval_mean_success import (
        q_learning_agent_eval_mean_reward_success_steps,
    )
    env = _RuleEvalEnv(indicator=(0,), reward=1.0, episode_len=2)
    out = q_learning_agent_eval_mean_reward_success_steps(_agent(), env, n_eval_episodes=3)
    assert math.isinf(out) and out < 0


# --------------------------------------------------------------------------- #
# Gating: a context rule (earlier column) firing does not count as the candidate
# --------------------------------------------------------------------------- #

def test_candidate_gated_by_earlier_context_rule():
    """With indicator (1, 1) a context rule (col 0) AND the candidate (col -1)
    both fire on the same step; the gating suppresses the candidate, so it reads
    as never-fired -> -inf."""
    from alpha_rule.policy_agents.q_learning.eval_total_reward import (
        q_learning_agent_eval_total_reward,
    )
    env = _RuleEvalEnv(indicator=(1, 1), reward=1.0, episode_len=2)
    score = q_learning_agent_eval_total_reward(_agent(), env, n_eval_episodes=2)
    assert math.isinf(score) and score < 0
