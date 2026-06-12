"""
Evaluator: ground-truth recognition distance.

Sibling of ``eval_recognition_distance`` that decouples the metric from the
Q-learning policy. Instead of reading whether the *trained agent* opened the
target chest (``otc_env.get_observations()["open"]``), this evaluator reads
``otc_env.boxes[box_index].is_ready()``, a ground-truth signal exposed by the
simulator that flips True whenever the activity's precondition is satisfied,
regardless of what the agent does. This avoids the ``-inf`` failure mode where
a correctly-firing rule scores poorly just because Q-learning didn't converge
to a working open-chest policy in 5,000 training steps.

Pairing strategy (asymmetric per-activity):
    - Collect every rising edge (0 to 1) of the gated rule-fire indicator and
      every rising edge of ``box.is_ready()`` across an episode.
    - For each ``is_ready`` rising edge, pair it with the nearest rule-fire
      rising edge (either preceding or following) and record ``-|Δsteps|``.
    - Average across all pairs over all episodes.

Gating: the rule-fire indicator uses the same ``any_previous`` guard as the
``rule`` mode, so a firing is suppressed if any earlier-discovered context
rule fired on the same step. This is what keeps the MCTS from re-learning a
rule that was already picked up in an earlier iteration of the outer discovery
loop.

Known limitations:
    - **Asymmetric over-fire trade-off**: a rule that fires on every step is
      *not* penalised; every activity still has a nearby firing. If this
      becomes a problem in practice, swap for a symmetric variant that also
      pairs each rule-fire with the nearest activity.
    - **Shared nearest**: when only one rule-fire occurs and multiple
      ``is_ready`` triggers happen far from it, all activities pair with the
      same firing, inflating the mean.
    - Still returns ``-np.inf`` if the rule never fires in any episode.
"""
from __future__ import annotations

import numpy as np

from alpha_rule.helpers.generic import find_attr_in_wrappers
from alpha_rule.policy_agents.q_learning.builder import get_state_tuple


def q_learning_agent_eval_ground_truth_distance(
    agent,
    env,
    box_index,
    n_eval_episodes=20,
):
    """
    Evaluate rule quality by step-distance between rule firings and the
    ground-truth ``is_ready()`` signal.

    Args:
        agent: tuple ``(q_table, policy)``, still used to step the env, but
            the metric does NOT depend on the agent's chest-opening actions.
        env: evaluation environment (must expose ``get_otc``).
        box_index: which box's ``is_ready()`` flag counts as the activity.
        n_eval_episodes: number of episodes to average.

    Returns:
        Mean of ``-|activity_step - nearest_recognition_step|`` over all
        (activity, nearest-rule-fire) pairs in all episodes, or ``-np.inf``
        if no pairs were produced.
    """
    # NOTE (asymmetric over-fire trade-off): this metric does NOT penalise
    # rules that over-fire. A rule that fires every step always has a nearby
    # fire for each is_ready event, so it scores well. If this becomes a
    # problem, switch to a symmetric variant: also pair each recognition_event
    # with the nearest activity_event and append ``-|delta|``.
    q_table, policy = agent
    distances = []

    for _ in range(n_eval_episodes):
        context_obs = env.reset()
        if isinstance(context_obs, tuple):
            context_obs = context_obs[0]
        done, truncated = False, False
        step = 0
        recognition_events: list[int] = []
        activity_events: list[int] = []
        prev_rule_gated = False
        prev_ready = False

        while not (done or truncated):
            full_state = get_state_tuple(env, context_obs)
            action = policy(full_state)

            # Gated rule-fire check: candidate is the LAST element of
            # context_obs; earlier elements are context rules already
            # discovered. Suppress firing if any context rule fired this step
            # (same guard as the "rule" mode in eval_recognition_distance).
            candidate_fires = bool(context_obs[-1] == 1)
            if len(context_obs) > 1:
                any_previous = bool(any(context_obs[:-1]))
                rule_gated = candidate_fires and not any_previous
            else:
                rule_gated = candidate_fires

            if rule_gated and not prev_rule_gated:
                recognition_events.append(step)
            prev_rule_gated = rule_gated

            context_obs, reward, done, truncated, *_ = env.step(action)
            if isinstance(context_obs, tuple):
                context_obs = context_obs[0]

            # Post-step activity check using the ground-truth readiness flag
            # (policy-independent, unlike ``get_observations()["open"]``).
            otc_env = find_attr_in_wrappers(env, "get_otc")()
            ready_now = bool(otc_env.boxes[box_index].is_ready())
            if ready_now and not prev_ready:
                activity_events.append(step)
            prev_ready = ready_now

            step += 1

        if recognition_events:
            for a in activity_events:
                nearest = min(recognition_events, key=lambda r: abs(r - a))
                distances.append(-abs(a - nearest))

    return np.mean(distances) if distances else -np.inf
