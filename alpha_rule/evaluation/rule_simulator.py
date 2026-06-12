"""
``RuleSimulator``: an ``Evaluator`` that scores a rule by training an RL agent.

Implements the ``Evaluator`` protocol by returning raw floats from
``evaluate``; the search loop wraps them in ``EvalResult`` when it needs the
richer surface.
"""
from __future__ import annotations

import gymnasium as gym

from alpha_rule.evaluation.evaluator import Evaluator


class RuleSimulator(Evaluator):
    """
    Score an MCTS rule node by training a small RL agent in the rule-wrapped
    environment and evaluating its policy.

    Pipeline:
        1. ``env = gym.make(env_name)``
        2. ``wrapped = transformer(env, rule_string)``
        3. ``agent = agent_builder(wrapped)``
        4. ``return agent_eval(agent, wrapped)``

    Args:
        env_name: gymnasium env id (e.g. ``'OpenTheChests-v0'``).
        agent_builder: callable taking the wrapped env and returning a
            trained agent. See ``policy_agents.q_learning.q_learning_agent_builder``.
        transformer: callable ``(env, rule_string) -> wrapped_env``. Usually
            one of the history-to-rule wrappers.
        agent_eval: callable ``(agent, env) -> float`` returning the
            scalar score for this rule.
    """

    def __init__(self, env_name, agent_builder, transformer, agent_eval):
        self.env_name = env_name
        self.agent_builder = agent_builder
        self.transformer = transformer
        self.agent_eval = agent_eval
        # Lazily-constructed base env reused across evaluate calls.
        # gym.make is expensive (env init, RNG seed, action space wiring);
        # paying it once per RuleSimulator instance instead of once per
        # evaluate call removes a constant overhead from every cache miss.
        # The wrapper chain is still freshly constructed per call so the
        # rolling event deque starts empty.
        self._env = None

    def _get_env(self):
        if self._env is None:
            self._env = gym.make(self.env_name)
        return self._env

    def evaluate(self, node):
        """
        Evaluate a rule node. Strips the trailing ``<END>`` marker from the
        node name so MCTS terminal nodes can be evaluated directly.
        """
        rule_str = node.name
        rule_str = rule_str.replace("<END>", "")
        env = self._get_env()
        transformed_env = self.transformer(env, rule_str)
        agent = self.agent_builder(transformed_env)
        return self.agent_eval(agent, transformed_env)
