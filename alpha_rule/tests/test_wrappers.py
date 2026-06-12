"""
Tests for the observation wrapper.

Uses the ``fake_gym_env`` fixture (from ``conftest``) to avoid depending
on real Gym / OpenTheChests. ``HistoryToRuleWrapperBase`` relies on three
env behaviours:
    - ``env.action_space.n`` at init
    - ``env.unwrapped.get_types()`` (preferred) or the ``.env`` chain
      returning the list of event types
    - ``reset`` / ``step`` returning dict observations with ``e_type`` /
      ``start`` / ``end``
``FakeGymEnv`` fulfils all three.

The base wrapper covers both single-rule and multi-rule use:
    - single-rule  == one rule string (str or list)
    - multi-rule   == ``strip_end_marker=True`` (strips a trailing
      ``<END>`` before parsing the rule)
"""
from __future__ import annotations

import numpy as np
import pytest

# The wrapper imports gymnasium at module load; skip the whole module when the
# optional [rl] extra is absent rather than erroring at collection.
pytest.importorskip("gymnasium")

from alpha_rule.wrappers.history_to_rule import HistoryToRuleWrapperBase  # noqa: E402


def _canned_events():
    """Events matching the pattern 'A then B' at the end of the rolling window."""
    return [
        {"e_type": 0, "start": np.array([0.0]), "end": np.array([1.0])},  # A
        {"e_type": 1, "start": np.array([5.0]), "end": np.array([6.0])},  # B
    ]


# --------------------------------------------------------------------------- #
# Observation shape: one binary entry per rule.
# --------------------------------------------------------------------------- #

def test_observation_shape_matches_rule_count(fake_gym_env):
    env = fake_gym_env(events=_canned_events(), event_types=("A", "B"))
    wrapped = HistoryToRuleWrapperBase(env, rule_list=["A", "B A <"])
    assert wrapped.observation_space.shape == (2,)


def test_single_rule_accepts_bare_string(fake_gym_env):
    """The base accepts a single string (single-rule convenience)
    and yields a length-1 binary vector."""
    env = fake_gym_env(events=_canned_events(), event_types=("A", "B"))
    wrapped = HistoryToRuleWrapperBase(env, rule_list="A")
    assert wrapped.observation_space.shape == (1,)
    obs = wrapped.observation(_canned_events()[0])
    assert obs.shape == (1,)


# --------------------------------------------------------------------------- #
# Rolling window trims past events.
# --------------------------------------------------------------------------- #

def test_rolling_window_trims(fake_gym_env):
    env = fake_gym_env(events=_canned_events(), event_types=("A", "B"))
    wrapped = HistoryToRuleWrapperBase(env, rule_list=["A"], window=3)
    # Emit more events than window; past events must be trimmed.
    for obs in _canned_events() * 3:
        wrapped.observation(obs)
    # The wrapper pops while ``len >= window``, so the list holds at most window-1.
    assert len(wrapped.passed_events_list) < wrapped.window


# --------------------------------------------------------------------------- #
# A matching rule -> 1, a non-matching rule -> 0.
# --------------------------------------------------------------------------- #

def test_detects_simple_last_event_rule(fake_gym_env):
    env = fake_gym_env(events=_canned_events(), event_types=("A", "B"))
    wrapped = HistoryToRuleWrapperBase(env, rule_list=["B"])
    # First observation is an A; rule "B" asks the last event to be B, so 0.
    first = wrapped.observation(_canned_events()[0])
    # Second observation is a B; the rule fires, so 1.
    second = wrapped.observation(_canned_events()[1])
    assert first[0] == 0
    assert second[0] == 1


# --------------------------------------------------------------------------- #
# reset clears the rolling window.
# --------------------------------------------------------------------------- #

def test_reset_clears_history(fake_gym_env):
    """
    ``reset`` zeroes the rolling window before ``super().reset()`` re-emits
    the first observation (which appends one event via ``observation``), so
    the post-reset list contains at most that single entry.
    """
    env = fake_gym_env(events=_canned_events(), event_types=("A", "B"))
    wrapped = HistoryToRuleWrapperBase(env, rule_list=["A"])
    # Grow the list past the post-reset size.
    wrapped.observation(_canned_events()[0])
    wrapped.observation(_canned_events()[1])
    assert len(wrapped.passed_events_list) == 2
    wrapped.reset()
    # After reset: the initial observation gets appended (at most 1 event).
    assert len(wrapped.passed_events_list) <= 1


# --------------------------------------------------------------------------- #
# strip_end_marker=True strips a trailing ``<END>`` before parsing the rule.
# --------------------------------------------------------------------------- #

def test_strip_end_marker_matches_terminated_rule(fake_gym_env):
    """
    With ``strip_end_marker=True`` a rule string carrying a trailing
    ``<END>`` parses identically to the same rule without it, so the
    terminated rule still fires on a matching history.
    """
    env = fake_gym_env(events=_canned_events(), event_types=("A", "B"))
    wrapped = HistoryToRuleWrapperBase(
        env, rule_list=["B <END>"], strip_end_marker=True,
    )
    # Last event B; the (stripped) rule "B" fires.
    wrapped.observation(_canned_events()[0])     # A
    second = wrapped.observation(_canned_events()[1])  # B
    assert second[0] == 1


def test_strip_end_marker_equivalent_to_unmarked_rule(fake_gym_env):
    """``"B <END>"`` (stripped) and ``"B"`` produce the same match vector."""
    env_a = fake_gym_env(events=_canned_events(), event_types=("A", "B"))
    env_b = fake_gym_env(events=_canned_events(), event_types=("A", "B"))
    stripped = HistoryToRuleWrapperBase(
        env_a, rule_list=["B <END>"], strip_end_marker=True,
    )
    plain = HistoryToRuleWrapperBase(env_b, rule_list=["B"])
    for ev in _canned_events():
        s = stripped.observation(ev)
        p = plain.observation(ev)
        assert s[0] == p[0]
