"""
Evaluator: mean reward, success rate, and mean episode length, or ``-inf``
when the candidate rule never fires across the eval episodes.

Column ordering: the candidate rule under evaluation must be the LAST entry of
the wrapper's context observation; earlier entries are context rules discovered
in previous outer-loop iterations. The ``-inf`` check reads ``context_obs[-1]``
on that assumption. Firing is read from the observation (the rule matching the
event history), not the agent's actions, so this ``-inf`` is policy-independent.
"""
from __future__ import annotations

import numpy as np

from alpha_rule.helpers.generic import find_attr_in_wrappers
from alpha_rule.policy_agents.q_learning.builder import get_state_tuple


def q_learning_agent_eval_mean_reward_success_steps(agent, env, n_eval_episodes=50):
    """
    Evaluate a Q-learning agent over ``n_eval_episodes``.

    Returns:
        ``(mean_reward, success_rate, mean_steps)`` where success is the
        fraction of episodes with total reward ≥ 1, OR ``-np.inf`` (a scalar)
        when the candidate rule never fired in any episode (see the module
        note on column ordering).
    """
    q_table, policy = agent
    rewards = []
    successes = 0
    steps_list = []
    candidate_seen = False

    # Resolve the wrapper-chain accessor once, not per step.
    get_otc = find_attr_in_wrappers(env, "get_otc")

    for _ in range(n_eval_episodes):
        context_obs = env.reset()
        if isinstance(context_obs, tuple):
            context_obs = context_obs[0]
        done, truncated = False, False
        total_reward = 0
        steps = 0

        while not (done or truncated):
            full_state = get_state_tuple(env, context_obs, get_otc=get_otc)
            action = policy(full_state)
            # The candidate rule is the last column of context_obs, gated so a
            # context rule firing this step does not count.
            if not candidate_seen:
                candidate_fires = bool(context_obs[-1] == 1)
                if len(context_obs) > 1:
                    fired = candidate_fires and not bool(any(context_obs[:-1]))
                else:
                    fired = candidate_fires
                if fired:
                    candidate_seen = True
            context_obs, reward, done, truncated, info = env.step(action)
            if isinstance(context_obs, tuple):
                context_obs = context_obs[0]
            total_reward += reward
            steps += 1

        rewards.append(total_reward)
        steps_list.append(steps)
        if total_reward >= 1:
            successes += 1

    # Structural failure: the rule never matched the history in any episode.
    if not candidate_seen:
        return float("-inf")
    mean_reward = np.mean(rewards)
    success_rate = successes / n_eval_episodes
    mean_steps = np.mean(steps_list)

    return mean_reward, success_rate, mean_steps
