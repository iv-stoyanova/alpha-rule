"""
Small, dependency-light helpers used across the package.

What lives here:
    - ``Event``: the canonical (type, start, end) triple used by every
      rule-matching routine.
    - ``event_obj_from_obs``: adapter from gymnasium-style dict observations
      to ``Event``.
    - ``find_attr_in_wrappers``: walks the ``.env`` chain looking for a
      named attribute (used by the Q-learning evaluators to reach
      ``get_otc``).
"""
from __future__ import annotations


class Event:
    """A single temporal event: ``(type, start, end)``."""

    __slots__ = ("type", "start", "end")

    def __init__(self, event_type, start, end):
        self.type = event_type
        self.start = start
        self.end = end

    def __repr__(self):
        return f"Event(type={self.type}, start={self.start}, end={self.end})"


def find_attr_in_wrappers(env, attr):
    """
    Recursively unwrap ``env`` until one of the layers has the attribute
    ``attr``; return the attribute.

    Raises:
        AttributeError: no wrapper in the chain exposes ``attr``.
    """
    while env:
        if hasattr(env, attr):
            return getattr(env, attr)
        if not hasattr(env, "env"):
            break
        env = env.env
    raise AttributeError(f"No wrapper has attribute '{attr}'")


def event_obj_from_obs(obs, all_event_types):
    """
    Build an ``Event`` from a gym-style dict observation.

    Args:
        obs: dict with ``e_type`` (int index), ``start`` (array[float]),
             ``end`` (array[float]).
        all_event_types: list mapping index to event-type string.

    Returns:
        Event instance.
    """
    return Event(
        event_type=all_event_types[obs["e_type"]],
        start=float(obs["start"][0]),
        end=float(obs["end"][0]),
    )
