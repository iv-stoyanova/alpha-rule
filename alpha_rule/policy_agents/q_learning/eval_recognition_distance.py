"""
Evaluator: recognition distance.

Measures how many steps separate a rule's firing from the true chest
opening. Lower-magnitude distances are better; returns the negated mean
absolute distance so MCTS can treat "larger is better" uniformly.

Filtering rules:
    - In "rule" mode we only count a recognition if no earlier rule
      (earlier columns in ``context_obs``) has already fired, which
      enforces temporal diversity between candidates.
    - In "action" mode we look at the agent's action for ``box_index``
      (supporting both array-valued and integer-encoded actions).

If either the rule never fires or the activity never occurs during the
episode, the episode is dropped from the mean.
"""
from __future__ import annotations

import numpy as np

from alpha_rule.helpers.generic import find_attr_in_wrappers
from alpha_rule.policy_agents.q_learning.builder import get_state_tuple


def q_learning_agent_eval_recognition_distance(
    agent,
    env,
    box_index,
    mode="rule",
    n_eval_episodes=20,
):
    """
    Evaluate rule quality by the step-distance between first recognition
    and the true activity.

    Args:
        agent: tuple ``(q_table, policy)``.
        env: evaluation environment (must expose ``get_otc``).
        box_index: which chest's opening counts as the true activity.
        mode: "rule" or "action"; see the module docstring.
        n_eval_episodes: number of episodes to average.

    Returns:
        Mean of ``-|activity_step - recognition_step|`` across episodes
        that saw both events, or ``-np.inf`` if none did.
    """
    assert mode in ("rule", "action"), f"mode must be 'rule' or 'action', got {mode!r}"
    q_table, policy = agent
    distances = []

    # Resolve the wrapper-chain accessor once for both get_state_tuple and the
    # per-step activity check.
    get_otc = find_attr_in_wrappers(env, "get_otc")

    for _ in range(n_eval_episodes):
        context_obs = env.reset()
        if isinstance(context_obs, tuple):
            context_obs = context_obs[0]
        done, truncated = False, False
        step = 0
        recognition_step = None
        activity_step = None

        while not (done or truncated):
            full_state = get_state_tuple(env, context_obs, get_otc=get_otc)
            action = policy(full_state)

            # Recognition check (pre-step): does the candidate rule fire?
            if recognition_step is None:
                if mode == "rule":
                    candidate_fires = bool(context_obs[-1] == 1)
                    if len(context_obs) > 1:
                        any_previous = bool(any(context_obs[:-1]))
                        fired = candidate_fires and not any_previous
                    else:
                        fired = candidate_fires
                else:  # "action"
                    if hasattr(action, "__len__"):
                        fired = bool(action[box_index])
                    else:
                        fired = bool((int(action) >> box_index) & 1)
                if fired:
                    recognition_step = step

            context_obs, reward, done, truncated, *_ = env.step(action)
            if isinstance(context_obs, tuple):
                context_obs = context_obs[0]

            # Post-step activity check: did the target box open this step?
            if activity_step is None:
                otc_env = get_otc()
                box_obs = otc_env.get_observations()["open"]
                # box_ground_truth = [box.is_ready() for box in otc_env.boxes]
                if box_obs[box_index]:
                    activity_step = step

            step += 1

        if recognition_step is not None and activity_step is not None:
            distances.append(-abs(activity_step - recognition_step))

    return np.mean(distances) if distances else -np.inf
