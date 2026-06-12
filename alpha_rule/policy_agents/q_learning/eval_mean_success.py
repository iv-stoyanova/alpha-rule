"""
Evaluator: mean reward, success rate, and mean episode length.

Used by the logging path in ``controlled_search`` to record all three
metrics for TensorBoard-style dashboards.
"""
from __future__ import annotations

import numpy as np

from alpha_rule.policy_agents.q_learning.builder import get_state_tuple


def q_learning_agent_eval_mean_reward_success_steps(agent, env, n_eval_episodes=50):
    """
    Evaluate a Q-learning agent over ``n_eval_episodes``.

    Returns:
        ``(mean_reward, success_rate, mean_steps)`` where success is the
        fraction of episodes with total reward ≥ 1.
    """
    q_table, policy = agent
    rewards = []
    successes = 0
    steps_list = []

    for _ in range(n_eval_episodes):
        context_obs = env.reset()
        if isinstance(context_obs, tuple):
            context_obs = context_obs[0]
        done, truncated = False, False
        total_reward = 0
        steps = 0

        while not (done or truncated):
            full_state = get_state_tuple(env, context_obs)
            action = policy(full_state)
            context_obs, reward, done, truncated, info = env.step(action)
            if isinstance(context_obs, tuple):
                context_obs = context_obs[0]
            total_reward += reward
            steps += 1

        rewards.append(total_reward)
        steps_list.append(steps)
        if total_reward >= 1:
            successes += 1

    mean_reward = np.mean(rewards)
    success_rate = successes / n_eval_episodes
    mean_steps = np.mean(steps_list)

    return mean_reward, success_rate, mean_steps
