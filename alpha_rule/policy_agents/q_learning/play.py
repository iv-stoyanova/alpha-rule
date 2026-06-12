"""
Debug helper: play a single episode with verbose per-step printing.
"""
from __future__ import annotations

from alpha_rule.policy_agents.q_learning.builder import get_state_tuple


def sample_agent_play_one_episode(agent, env, render=False):
    """
    Run one episode, printing observations, actions, rewards at every step.

    Args:
        agent: callable mapping a full state tuple to an integer action.
        env: Gym environment.
        render: call ``env.render()`` if True.
    """
    context_obs = env.reset()
    if isinstance(context_obs, tuple):
        context_obs = context_obs[0]

    done, truncated = False, False
    total_reward = 0
    step = 0

    while not (done or truncated):
        full_state = get_state_tuple(env, context_obs)
        action = agent(full_state)

        if render:
            env.render()

        print(f"Step {step}:")
        print(f"  Context state: {context_obs}")
        print(f"  Box state: {full_state[len(context_obs):]}")
        print(f"  Action taken: {action}")

        context_obs, reward, done, truncated, info = env.step(action)
        if isinstance(context_obs, tuple):
            context_obs = context_obs[0]

        print(f"  Reward received: {reward}")
        print(f"  Done: {done}\n")

        total_reward += reward
        step += 1

    print(f"Episode finished. Total reward: {total_reward}")
