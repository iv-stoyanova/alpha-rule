"""
Evaluator: total episode reward, bonused when the agent reaches a
"seen-once" marker state (``full_state[0] == 1``).

Used by the reward-shaping MCTS loops where simply finding a matching
rule should be worth extra signal.
"""
from __future__ import annotations

import numpy as np

from alpha_rule.policy_agents.q_learning.builder import get_state_tuple


def q_learning_agent_eval_total_reward(agent, env, added_reward=50, n_eval_episodes=20):
    """
    Evaluate a Q-learning agent by its average episodic reward.

    Args:
        agent: tuple ``(q_table, policy)`` returned by the builder.
        env: evaluation environment.
        added_reward: bonus added when the agent ever reaches ``full_state[0] == 1``.
        n_eval_episodes: number of episodes.

    Returns:
        ``mean_reward + added_reward`` if the marker state was seen at
        least once; ``-np.inf`` otherwise.
    """
    q_table, agent_policy = agent
    seen_once = False
    rewards = []

    for _ in range(n_eval_episodes):
        context_obs = env.reset()
        if isinstance(context_obs, tuple):
            context_obs = context_obs[0]
        done, truncated = False, False
        total_reward = 0

        while not (done or truncated):
            full_state = get_state_tuple(env, context_obs)
            action = agent_policy(full_state)
            if not seen_once and full_state[0] == 1:
                seen_once = True
            context_obs, reward, done, truncated, *_ = env.step(action)
            if isinstance(context_obs, tuple):
                context_obs = context_obs[0]
            total_reward += reward

        rewards.append(total_reward)

    mean_reward = np.mean(rewards)
    if seen_once:
        return added_reward + mean_reward
    return -np.inf
