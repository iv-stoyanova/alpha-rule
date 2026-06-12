"""
Tests for ``alpha_rule.evaluation.rule_simulator.RuleSimulator``.

Pins the env-reuse contract: a single ``gym.make`` call per
``RuleSimulator`` instance, regardless of how many ``evaluate`` calls
are made. Saves a real per-call constant cost on every cache miss.
"""
from __future__ import annotations

from dataclasses import dataclass

import pytest

# RuleSimulator imports gymnasium at module load; skip the whole module when
# the optional [rl] extra is absent rather than erroring at collection.
pytest.importorskip("gymnasium")


@dataclass
class _FakeNode:
    name: str


class _SentinelEnv:
    """Stand-in for a gym env; tracks reset/step calls. Not a real Env."""

    def __init__(self):
        self.resets = 0
        self.steps = 0

    def reset(self):
        self.resets += 1
        return None

    def step(self, action):
        self.steps += 1
        return None, 0.0, True, False, {}


class _SentinelWrapper:
    """Stand-in for the transformer's output; records the underlying env."""

    def __init__(self, env, rule_str):
        self.env = env
        self.rule_str = rule_str


def _patch_gym_make(rs_module, fake_make):
    """Return a context-manager-like (no-yield) tuple ``(orig, restore)``.

    Used because the custom test runner doesn't ship ``monkeypatch``.
    """
    orig = rs_module.gym.make

    def restore():
        rs_module.gym.make = orig

    rs_module.gym.make = fake_make
    return restore


def test_rule_simulator_reuses_env_across_evaluate_calls():
    """
    The env is constructed lazily and reused. Two evaluate calls hit
    the cached env and only one ``gym.make`` happens overall.
    """
    from alpha_rule.evaluation import rule_simulator as rs

    make_calls = []

    def fake_make(env_name):
        env = _SentinelEnv()
        make_calls.append((env_name, env))
        return env

    restore = _patch_gym_make(rs, fake_make)
    try:
        transformer_calls = []

        def transformer(env, rule_str):
            transformer_calls.append((id(env), rule_str))
            return _SentinelWrapper(env, rule_str)

        def builder(wrapped):
            return ("agent", id(wrapped.env))

        def evaluator(agent, wrapped):
            return 1.0

        sim = rs.RuleSimulator(
            env_name="fake-v0",
            agent_builder=builder,
            transformer=transformer,
            agent_eval=evaluator,
        )

        sim.evaluate(_FakeNode(name="A"))
        sim.evaluate(_FakeNode(name="A < B"))

        assert len(make_calls) == 1, \
            f"gym.make should be called once; got {len(make_calls)}"

        # Both transformer calls received the same env instance.
        assert transformer_calls[0][0] == transformer_calls[1][0]
    finally:
        restore()


def test_rule_simulator_constructs_env_lazily():
    """No ``gym.make`` until the first ``evaluate`` call."""
    from alpha_rule.evaluation import rule_simulator as rs

    calls = []

    def fake_make(name):
        calls.append(name)
        return _SentinelEnv()

    restore = _patch_gym_make(rs, fake_make)
    try:
        sim = rs.RuleSimulator(
            env_name="fake-v0",
            agent_builder=lambda e: None,
            transformer=lambda env, r: _SentinelWrapper(env, r),
            agent_eval=lambda a, e: 0.0,
        )
        assert sim._env is None
        assert calls == []

        sim.evaluate(_FakeNode(name="A"))
        assert sim._env is not None
        assert len(calls) == 1
    finally:
        restore()


def test_rule_simulator_strips_end_marker():
    """Terminal nodes have a trailing ``<END>`` that must be stripped
    before the rule string reaches the transformer."""
    from alpha_rule.evaluation import rule_simulator as rs

    seen_rules = []

    restore = _patch_gym_make(rs, lambda n: _SentinelEnv())
    try:
        def transformer(env, rule_str):
            seen_rules.append(rule_str)
            return _SentinelWrapper(env, rule_str)

        sim = rs.RuleSimulator(
            env_name="fake-v0",
            agent_builder=lambda e: None,
            transformer=transformer,
            agent_eval=lambda a, e: 0.0,
        )

        sim.evaluate(_FakeNode(name="A <END>"))
        assert "<END>" not in seen_rules[0]
    finally:
        restore()
