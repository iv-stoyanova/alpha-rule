"""
Tabular Q-learning agent builder and shared state-tuple helper.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np

from alpha_rule.helpers.generic import find_attr_in_wrappers


def get_state_tuple(env, context_obs, get_otc=None):
    """
    Build a composite state tuple from the rule-wrapper context observation
    and the unwrapped OTC environment's box observation.

    Args:
        env: the (possibly nested) Gym environment.
        context_obs: the observation coming out of the outer wrapper.
        get_otc: optional pre-resolved ``get_otc`` accessor (skips the
            wrapper-chain walk on each call). When omitted, falls back to
            ``find_attr_in_wrappers(env, "get_otc")``.

    Returns:
        Tuple concatenating context bits with the box-open bits.
    """
    if get_otc is None:
        get_otc = find_attr_in_wrappers(env, "get_otc")
    box_obs = get_otc().get_observations()["open"]
    return tuple(context_obs) + tuple(box_obs)


def _q_table_get(q_table, state, n_actions):
    """Lookup with implicit zero init. Replaces ``defaultdict`` to skip the
    lambda invocation per miss (small but per-step hot path)."""
    arr = q_table.get(state)
    if arr is None:
        arr = np.zeros(n_actions)
        q_table[state] = arr
    return arr


def q_learning_agent_builder(
        env,
        total_timesteps=1_000,
        alpha=0.1,
        gamma=0.97,
        epsilon=0.1,
        early_stop_tol=0.0,
        check_interval=200,
        patience=3,
        seed=None,
):
    """
    Build and train a tabular Q-learning agent.

    Args:
        env: Gym environment with ``Discrete`` action space.
        total_timesteps: number of steps to run ε-greedy Q-learning.
            Defaults to 1,000. Tabular Q-learning on the OTC-sized state
            space typically converges well below this budget. Override
            explicitly for harder envs.
        alpha: learning rate.
        gamma: discount factor.
        epsilon: ε in ε-greedy exploration.
        early_stop_tol: if > 0, break the training loop early once the
            Q-table has stopped changing meaningfully. Every
            ``check_interval`` steps the max absolute delta across visited
            states is compared against this tolerance; if it stays below
            for ``patience`` consecutive checks the loop exits. Default
            ``0.0`` disables (preserves the pre-change behaviour).
        check_interval: steps between convergence checks. Ignored when
            ``early_stop_tol == 0``.
        patience: number of consecutive sub-tolerance checks required to
            exit. Ignored when ``early_stop_tol == 0``.
        seed: if given, makes training reproducible: ε-greedy draws use a
            local ``np.random.Generator(seed)`` and the action space is seeded
            for ``sample()``. ``None`` uses the global ``np.random``.

    Returns:
        ``(q_table, policy)`` where ``policy(full_state)`` returns an int
        action via greedy argmax.
    """
    assert isinstance(env.action_space, gym.spaces.Discrete), "Only discrete actions supported"

    # Action set is taken from the env's action space. Wrap the env in
    # ``OneHotBoxActionWrapper`` upstream to get the "open one box or none"
    # space (size n_boxes + 1); without it this is the raw button mask.
    n_actions = env.action_space.n
    q_table: dict = {}

    # Local RNG when seeded, else the global np.random. Both expose .random().
    rng = np.random.default_rng(seed) if seed is not None else np.random
    if seed is not None:
        env.action_space.seed(seed)

    # Resolve the wrapper-chain accessor once. ``find_attr_in_wrappers``
    # walks ``.env`` recursively, which is wasted work to repeat every step.
    get_otc = find_attr_in_wrappers(env, "get_otc")

    context_obs = env.reset()
    if isinstance(context_obs, tuple):
        context_obs = context_obs[0]

    prev_snapshot = None
    stable_count = 0

    for t in range(total_timesteps):
        state = get_state_tuple(env, context_obs, get_otc=get_otc)
        state_q = _q_table_get(q_table, state, n_actions)

        if rng.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = int(np.argmax(state_q))

        # Step with the agent's action directly; any one-hot/box remapping is
        # owned by an upstream action wrapper (OneHotBoxActionWrapper), not by
        # this builder, so the same action is interpreted identically in
        # training and evaluation.
        next_obs, reward, done, truncated, *_ = env.step(action)
        # Open-chest bonus: amplify the positive reward so the agent is strongly
        # motivated to open a box rather than play it safe.
        if reward > 0:
            reward += 10
        if isinstance(next_obs, tuple):
            next_obs = next_obs[0]

        # TD target: bootstrap gamma*max(next) only on non-terminal transitions.
        # A terminal step has no successor, so its target is the reward alone.
        # Truncation is treated the same way, since the state carries no time
        # feature.
        # print(f"reward: {reward} ")
        if done:
            # print("all chest open")
            pass
        if truncated:
            # print("truncated")
            pass
        if done or truncated:
            td_target = reward
        else:
            next_state = get_state_tuple(env, next_obs, get_otc=get_otc)
            next_state_q = _q_table_get(q_table, next_state, n_actions)
            td_target = reward + gamma * np.max(next_state_q)
        state_q[action] += alpha * (td_target - state_q[action])

        if not (done or truncated):
            context_obs = next_obs
        else:
            reset = env.reset()
            context_obs = reset[0] if isinstance(reset, tuple) else reset

        if early_stop_tol > 0.0 and (t + 1) % check_interval == 0:
            current_snapshot = {s: arr.copy() for s, arr in q_table.items()}
            if prev_snapshot is not None:
                # Max abs delta across the union of visited states. New
                # states present only in ``current_snapshot`` compare
                # against an implicit-zero vector, so freshly-touched
                # states count as full-magnitude changes.
                deltas = []
                for s, arr in current_snapshot.items():
                    prev_arr = prev_snapshot.get(s)
                    if prev_arr is None:
                        deltas.append(float(np.max(np.abs(arr))))
                    else:
                        deltas.append(float(np.max(np.abs(arr - prev_arr))))
                delta = max(deltas) if deltas else 0.0
                if delta < early_stop_tol:
                    stable_count += 1
                    if stable_count >= patience:
                        break
                else:
                    stable_count = 0
            prev_snapshot = current_snapshot

    def policy(full_state):
        arr = q_table.get(full_state)
        if arr is None:
            return 0
        return int(np.argmax(arr))

    # print(policy)
    # print(q_table)
    # print("a"+5)
    return q_table, policy
