"""
``RuleSimulator``: an ``Evaluator`` that scores a rule by training an RL agent.

Implements the ``Evaluator`` protocol by returning raw floats from
``evaluate``; the search loop wraps them in ``EvalResult`` when it needs the
richer surface.
"""
from __future__ import annotations

import random
import warnings
from typing import Optional

import gymnasium as gym

from alpha_rule.evaluation.evaluator import Evaluator


class RuleSimulator(Evaluator):
    """
    Score an MCTS rule node by training a small RL agent in the rule-wrapped
    environment and evaluating its policy.

    Pipeline:
        1. ``env = gym.make(env_name)``
        2. ``wrapped = transformer(env, rule_string)``
        3. ``agent = agent_builder(wrapped, **agent_builder_kwargs)``
        4. ``return agent_eval(agent, wrapped)``

    Args:
        env_name: gymnasium env id (e.g. ``'OpenTheChests-v0'``).
        agent_builder: callable taking the wrapped env (and any
            ``agent_builder_kwargs``) and returning a trained agent. See
            ``policy_agents.q_learning.q_learning_agent_builder``.
        transformer: callable ``(env, rule_string) -> wrapped_env``. Usually
            one of the history-to-rule wrappers.
        agent_eval: callable ``(agent, env) -> float`` returning the
            scalar score for this rule.
        reward_scale: positive reward cap used to scale value targets
            (``z = clip(value / reward_scale, -1, 1)``). ``None`` (default)
            auto-derives it from the env's box count on first use:
            ``num_boxes`` is the positive cap because the agent earns at most
            ``+1`` per box (episodic reward ranges ``[-max_steps, num_boxes]``,
            the negative tail clips to ``-1``). This matches the reward range of
            both ``q_learning_agent_eval_mean_reward_success_steps`` and
            ``q_learning_agent_eval_total_reward``. Pass a float to override.
        agent_builder_kwargs: optional dict forwarded to ``agent_builder`` on
            every ``evaluate``. A fresh agent is trained per leaf, so this is
            where to cut that cost, e.g.
            ``{"total_timesteps": 500, "early_stop_tol": 1e-3}``.
        seed: if given, the shared base env is re-seeded at the start of each
            ``evaluate`` so a rule scores reproducibly instead of inheriting the
            RNG position left by the previous call. For fully reproducible
            training also pass ``seed`` via ``agent_builder_kwargs`` (the
            Q-learning builder accepts it).
        resample_seed: when ``True``, draw a fresh seed from a ``seed``-seeded
            RNG on each ``evaluate`` (when no explicit ``seed`` is passed),
            retrain the agent with it, and eval it, so every call is an
            independent, reproducible draw from the rule's return distribution.
            Default ``False`` (``seed=None`` resets the env to ``self.seed`` and
            trains with ``agent_builder_kwargs`` unchanged: deterministic per
            rule).
    """

    def __init__(
        self,
        env_name,
        agent_builder,
        transformer,
        agent_eval,
        *,
        reward_scale: Optional[float] = None,
        agent_builder_kwargs: Optional[dict] = None,
        seed: Optional[int] = None,
        resample_seed: bool = False,
    ):
        self.env_name = env_name
        self.agent_builder = agent_builder
        self.transformer = transformer
        self.agent_eval = agent_eval
        # ``reward_scale`` is a property: an explicit value wins; otherwise it is
        # derived once from the env's box count (see the property below).

        self._reward_scale = reward_scale
        self._reward_scale_resolved = reward_scale is not None
        self.agent_builder_kwargs = dict(agent_builder_kwargs or {})
        self.seed = seed
        self.resample_seed = resample_seed
        # Lazily built RNG, seeded from ``seed``, that draws a fresh per-call seed
        # under resample_seed so each evaluate is an independent reproducible draw.
        self._seed_rng: Optional[random.Random] = None
        # Lazily-constructed base env reused across evaluate calls.
        # gym.make is expensive (env init, RNG seed, action space wiring);
        # paying it once per RuleSimulator instance instead of once per
        # evaluate call removes a constant overhead from every cache miss.
        # The wrapper chain is still freshly constructed per call so the
        # rolling event deque starts empty.
        self._env = None
        self._warned_no_scale = False

    @property
    def reward_scale(self):
        """Positive reward cap used by the training stack to scale value
        targets (``z = clip(value / reward_scale, -1, 1)``).

        When not set explicitly it is derived once from the environment's box
        count: the agent earns at most ``+1`` per box, so the episodic reward
        ranges roughly ``[-max_steps, num_boxes]`` and ``num_boxes`` is the
        positive cap (the negative tail clips to ``-1``). Reading this property
        builds the cached env if needed. ``None`` only if the box count could
        not be read, in which case a one-time warning fires and value_scale
        falls back to ``1.0``."""
        if not self._reward_scale_resolved:
            self._reward_scale_resolved = True
            self._reward_scale = self._derive_reward_scale_from_num_boxes()
        return self._reward_scale

    @reward_scale.setter
    def reward_scale(self, value):
        self._reward_scale = value
        self._reward_scale_resolved = True

    def _derive_reward_scale_from_num_boxes(self):
        """Read ``num_boxes`` off the OTC env (the positive reward cap)."""
        from alpha_rule.helpers.generic import find_attr_in_wrappers
        try:
            env = self._get_env()
            env.reset()                       # populate the boxes
            otc = find_attr_in_wrappers(env, "get_otc")()
            boxes = getattr(otc, "boxes", None)
            if boxes is not None and len(boxes) > 0:
                return float(len(boxes))
        except Exception:
            pass
        if not self._warned_no_scale:
            warnings.warn(
                "RuleSimulator could not derive reward_scale from the env's "
                "num_boxes; the training stack will scale value targets by 1.0, "
                "which saturates them when agent_eval returns values outside "
                "[-1, 1]. Pass reward_scale explicitly.",
                stacklevel=3,
            )
            self._warned_no_scale = True
        return None

    def _get_env(self):
        if self._env is None:
            self._env = gym.make(self.env_name)
        return self._env

    def _next_seed(self) -> int:
        """Draw a fresh seed from the ``seed``-seeded RNG, so each evaluate under
        resample_seed retrains and evals on a new, reproducible instantiation."""
        if self._seed_rng is None:
            self._seed_rng = random.Random(self.seed)
        return self._seed_rng.randrange(2 ** 31 - 1)

    def evaluate(self, node, *, seed=None):
        """
        Evaluate a rule node. Strips the trailing ``<END>`` marker from the
        node name so MCTS terminal nodes can be evaluated directly.

        Args:
            seed: optional per-call seed overriding ``self.seed`` for this
                evaluation; re-seeds the env and the agent builder so repeated
                calls with distinct seeds are independent samples. ``None``
                (default) resets the env to ``self.seed`` and trains with
                ``agent_builder_kwargs`` unchanged, except under
                ``resample_seed`` where a fresh seed is drawn from the seeded RNG
                so each call retrains and evals on a new, reproducible draw.
        """
        # Resolve reward_scale on first use even if no consumer read it at setup.
        _ = self.reward_scale

        rule_str = node.name.replace("<END>", "")
        env = self._get_env()

        if self.resample_seed and seed is None:
            # Fresh per-call seed: full retrain + eval, varied but reproducible.
            effective_seed = self._next_seed()
            builder_kwargs = {**self.agent_builder_kwargs, "seed": effective_seed}
        else:
            effective_seed = self.seed if seed is None else seed
            # Override the builder seed only when a per-call seed was given.
            builder_kwargs = (
                self.agent_builder_kwargs
                if seed is None
                else {**self.agent_builder_kwargs, "seed": seed}
            )
        if effective_seed is not None:
            # Re-seed the base env so this rule scores reproducibly.
            env.reset(seed=effective_seed)
        transformed_env = self.transformer(env, rule_str)
        agent = self.agent_builder(transformed_env, **builder_kwargs)
        return self.agent_eval(agent, transformed_env)
