"""
Evaluator: mean total episode reward, or ``-inf`` when the candidate rule
never fires across the eval episodes.

The candidate rule is the LAST column of the context observation (earlier
columns are context rules discovered in previous outer-loop iterations); a
firing is gated so a previously-discovered context rule firing this step does
not count. A rule that never fires earns ``-inf`` (a structural failure),
matching the ``-inf`` convention used elsewhere in the backend.
"""
from __future__ import annotations

import numpy as np

from alpha_rule.helpers.generic import find_attr_in_wrappers
from alpha_rule.policy_agents.q_learning.builder import get_state_tuple


def q_learning_agent_eval_total_reward(agent, env, n_eval_episodes=20):
    """
    Evaluate a Q-learning agent by its average episodic reward.

    Args:
        agent: tuple ``(q_table, policy)`` returned by the builder.
        env: evaluation environment.
        n_eval_episodes: number of episodes.

    Returns:
        ``mean_reward`` if the candidate rule fired in at least one episode;
        ``-np.inf`` otherwise.
    """
    q_table, agent_policy = agent
    seen_once = False
    rewards = []

    # Resolve the wrapper-chain accessor once, not per step.
    get_otc = find_attr_in_wrappers(env, "get_otc")

    for _ in range(n_eval_episodes):
        context_obs = env.reset()
        if isinstance(context_obs, tuple):
            context_obs = context_obs[0]
        done, truncated = False, False
        total_reward = 0

        while not (done or truncated):
            full_state = get_state_tuple(env, context_obs, get_otc=get_otc)
            action = agent_policy(full_state)
            # The candidate rule is the last column of context_obs, gated so a
            # context rule firing this step does not count.
            if not seen_once:
                candidate_fires = bool(context_obs[-1] == 1)
                if len(context_obs) > 1:
                    fired = candidate_fires and not bool(any(context_obs[:-1]))
                else:
                    fired = candidate_fires
                if fired:
                    seen_once = True
            context_obs, reward, done, truncated, *_ = env.step(action)
            if isinstance(context_obs, tuple):
                context_obs = context_obs[0]
            total_reward += reward

        rewards.append(total_reward)

    mean_reward = np.mean(rewards)
    if seen_once:
        return float(mean_reward)
    return -np.inf
