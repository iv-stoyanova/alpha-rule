"""
History-to-rule observation wrapper.

``HistoryToRuleWrapperBase`` maintains a rolling window of the most recent
events, checks one or more Allen-rule matrices against that window, and
emits a binary indicator vector (one entry per rule).

Two behaviours fold in via constructor args (they were two separate
classes before): pass a single rule string *or* a list of them, and set
``strip_end_marker=True`` to strip trailing ``<END>`` tokens from rule
strings before parsing. Event types resolve via
``env.unwrapped.get_types()`` with a tolerant fallback down the
``.env`` chain.
"""
from __future__ import annotations

from collections import deque
from typing import Iterable, List, Optional, Sequence

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from alpha_rule.helpers.generic import event_obj_from_obs
from alpha_rule.rules.allen_matrix import AllenMatrix
from alpha_rule.rules.rule_matching import (
    determine_allen_relation,
    match_rule_to_history,
    match_rule_to_matrix,
)


def _as_rule_list(rules: object) -> List[str]:
    """
    Accept either a list of rule strings or a single string.

    A single rule string is accepted directly as a convenience.
    """
    if isinstance(rules, str):
        return [rules]
    return list(rules)


class HistoryToRuleWrapperBase(gym.ObservationWrapper):
    """
    ObservationWrapper that replaces the env observation with a binary
    vector indicating which of ``rule_list`` currently matches the recent
    event history.

    Args:
        env: base gym / gymnasium environment. Must expose ``get_types()``
             somewhere on the unwrap chain.
        rule_list: iterable of Allen hierarchy strings; a single string is
                   accepted and wrapped.
        window: maximum number of recent events retained in the rolling
                buffer. Older events are discarded.
        strip_end_marker: if True, trailing ``<END>`` tokens are stripped
                          from each rule string before parsing, so a
                          terminated rule parses the same as its unmarked form.
    """

    def __init__(
        self,
        env,
        rule_list: Iterable[str] | str,
        window: int = 15,
        *,
        strip_end_marker: bool = False,
    ):
        super().__init__(env)
        self.rule_list: List[str] = _as_rule_list(rule_list)

        parsed = []
        for rule_str in self.rule_list:
            cleaned = rule_str.replace("<END>", "") if strip_end_marker else rule_str
            parsed.append(AllenMatrix.from_hierarchy_string(cleaned))
        self.all_rule_matrices: List[AllenMatrix] = parsed

        self.action_dim = len(bin(env.action_space.n)) - 3
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(len(self.rule_list),), dtype=np.int8,
        )
        self.all_event_types: Sequence[str] = self._resolve_event_types(env)
        self.window = window
        # Rolling window backed by deque for O(1) trim. Legacy code did
        # list.pop(0) (O(n) per step); deque.popleft makes the wrapper's
        # observation hot-path linear in the number of rules, not window size.
        # `passed_events_list` is exposed as a property for back-compat so
        # tests and external code can read it as if it were a list.
        self._events: deque = deque()
        # Incrementally maintained history matrix (raw numpy, no AllenMatrix
        # wrapper). Shape ``(n+2, n)`` with newest event at col 0. Updated on
        # every ``observation()`` and consumed by ``match_rule_to_matrix`` to
        # skip the per-step O(window²) Allen-relation rebuild. ``None`` until
        # the first event arrives.
        self._history_matrix: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ #
    # Event-type lookup: tolerant of wrapping depth.
    # ------------------------------------------------------------------ #

    @staticmethod
    def _resolve_event_types(env) -> Sequence[str]:
        """
        Try ``env.unwrapped.get_types()`` first (preferred). Fall back to
        walking the ``.env`` chain manually if it is not reachable there.
        """
        unwrapped = getattr(env, "unwrapped", None)
        if unwrapped is not None and hasattr(unwrapped, "get_types"):
            return unwrapped.get_types()
        # Fallback: walk `.env` manually.
        probe = env
        for _ in range(6):
            if hasattr(probe, "get_types"):
                return probe.get_types()
            if not hasattr(probe, "env"):
                break
            probe = probe.env
        raise AttributeError("env does not expose get_types()")

    # ------------------------------------------------------------------ #
    # Back-compat accessor: external code reads this as a list.
    # ------------------------------------------------------------------ #

    @property
    def passed_events_list(self):
        return self._events

    @passed_events_list.setter
    def passed_events_list(self, value):
        # Back-compat: assignment to the old list name is honoured. Wrap
        # incoming iterables in a fresh deque so trim semantics survive.
        self._events = deque(value)

    # ------------------------------------------------------------------ #
    # ObservationWrapper interface
    # ------------------------------------------------------------------ #

    # ------------------------------------------------------------------ #
    # Incremental matrix maintenance
    # ------------------------------------------------------------------ #

    def _append_event_to_matrix(self, new_event):
        """Update ``self._history_matrix`` in-place to include ``new_event``
        at column 0 (the newest-event column). O(n_old) Allen-relation
        computations vs O(n²) for a full rebuild."""
        # Invariant behind the all-ones indicator shortcut used below: event
        # types are never the matrix fillers '#'/'='. If that ever changes the
        # shortcut would write a wrong indicator row, so guard it cheaply.
        assert new_event.type not in ("#", "="), (
            f"history event type {new_event.type!r} collides with a matrix "
            "filler; the all-ones indicator shortcut is invalid"
        )
        if self._history_matrix is None or self._history_matrix.shape[1] == 0:
            m = np.full((3, 1), "#", dtype=object)
            m[1, 0] = new_event.type
            m[2, 0] = "="
            # Indicator is always all-ones (a real event type in row 1 means the
            # single column is never removable), so skip the per-step
            # check_rows_columns_combined (np.isin over an object array).
            m[0] = np.ones(1, dtype=int)
            self._history_matrix = m
            return

        old = self._history_matrix
        n_old = old.shape[1]
        n_new = n_old + 1

        new = np.full((n_new + 2, n_new), "#", dtype=object)

        # Row 1 (types): newest first.
        new[1, 0] = new_event.type
        new[1, 1:] = old[1, :]

        # Existing relations shift diagonally: OLD[r, c] -> NEW[r+1, c+1].
        if n_old > 0:
            new[3:n_new + 2, 1:n_new] = old[2:n_old + 2, 0:n_old]

        # Relations between new_event (newest, col 0) and each older
        # event. NEW[2, j] = relation(old_history[n_new-1-j], new_event)
        # for j=1..n_old. In self._events (oldest first), the event at
        # position n_old - j is the same as old_history[n_old - j].
        for j in range(1, n_new):
            old_event = self._events[n_old - j]
            new[2, j] = determine_allen_relation(old_event, new_event)

        # Diagonal "=" (overwrites the slice copy at diagonal positions).
        for i in range(n_new):
            new[i + 2, i] = "="

        # Always all-ones under the real-type invariant (see the assert above),
        # so skip the object-array check_rows_columns_combined recompute.
        new[0] = np.ones(n_new, dtype=int)
        self._history_matrix = new

    def _trim_oldest_from_matrix(self):
        """Drop the rightmost column (oldest event) and its relation row."""
        if self._history_matrix is None:
            return
        n = self._history_matrix.shape[1]
        if n <= 1:
            self._history_matrix = None
            return
        trimmed = self._history_matrix[:n + 1, :n - 1].copy()
        # Dropping the oldest column leaves real types in row 1, so the
        # indicator stays all-ones; no recompute needed.
        trimmed[0] = np.ones(n - 1, dtype=int)
        self._history_matrix = trimmed

    # ------------------------------------------------------------------ #
    # ObservationWrapper interface
    # ------------------------------------------------------------------ #

    def observation(self, obs):
        event = event_obj_from_obs(obs, self.all_event_types)
        self._append_event_to_matrix(event)
        self._events.append(event)
        # Trim down to exactly ``window`` events.
        while len(self._events) > self.window:
            self._events.popleft()
            self._trim_oldest_from_matrix()

        rule_validation = np.zeros(len(self.rule_list))
        for ind, rule_matrix in enumerate(self.all_rule_matrices):
            rule_validation[ind] = match_rule_to_matrix(
                rule_matrix, self._history_matrix,
            )
        return rule_validation

    def reset(self, **kwargs):
        self._events = deque()
        self._history_matrix = None
        # Forward seed/options and return the standard gym ``(obs, info)``
        # 2-tuple. ``ObservationWrapper.reset`` already applies ``observation()``.
        return super().reset(**kwargs)
