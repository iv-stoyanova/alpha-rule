"""
Tests for the incrementally-maintained Allen matrix on
``HistoryToRuleWrapperBase``.

Pins:
    - After each event arrives, the incremental matrix equals what
      ``generate_allen_matrix_from_history`` would produce on the same
      current window.
    - Indicator row is recomputed correctly after every append and trim.
    - ``match_rule_to_matrix`` returns the same boolean as
      ``match_rule_to_history`` on every step.
    - Wrapper trim semantics hold (deque length matches the window bound).
"""
from __future__ import annotations

import numpy as np
import pytest

# The wrapper (imported inside the tests) pulls in gymnasium; skip the whole
# module when the optional [rl] extra is absent rather than erroring.
pytest.importorskip("gymnasium")

from alpha_rule.helpers.generic import Event  # noqa: E402
from alpha_rule.rules.allen_matrix import AllenMatrix
from alpha_rule.rules.rule_matching import (
    generate_allen_matrix_from_history,
    match_rule_to_history,
    match_rule_to_matrix,
)


def _make_event(t, start, end):
    return Event(event_type=t, start=start, end=end)


def _step_matrix(events_so_far, window):
    """Replicate the wrapper's window logic: append events one at a
    time, trimming when the deque hits ``window``."""
    from collections import deque
    d = deque()
    for e in events_so_far:
        d.append(e)
        while len(d) >= window:
            d.popleft()
    return list(d)


def test_incremental_matrix_matches_full_rebuild_step_by_step():
    """Drive the wrapper with N events one at a time; after each event
    the incremental matrix must equal a fresh rebuild on the same
    window."""
    # Build the wrapper logic in isolation. We don't need a real Gym env
    # for this; we only exercise the incremental update functions.
    from alpha_rule.wrappers.history_to_rule import HistoryToRuleWrapperBase

    events = [
        _make_event("A", 0, 2),
        _make_event("B", 3, 5),
        _make_event("A", 6, 8),
        _make_event("C", 9, 11),
        _make_event("B", 12, 14),
        _make_event("A", 15, 17),
    ]

    # Construct a minimal wrapper-like state holder. We can't easily
    # instantiate HistoryToRuleWrapperBase without a Gym env, so reach
    # into its private helpers via a stand-in.
    class _StandIn:
        _append_event_to_matrix = HistoryToRuleWrapperBase._append_event_to_matrix
        _trim_oldest_from_matrix = HistoryToRuleWrapperBase._trim_oldest_from_matrix

        def __init__(self):
            from collections import deque
            self._events = deque()
            self._history_matrix = None
            self.window = 4

    s = _StandIn()
    for i, ev in enumerate(events):
        s._append_event_to_matrix(s, ev) if False else s._append_event_to_matrix(ev)
        s._events.append(ev)
        while len(s._events) >= s.window:
            s._events.popleft()
            s._trim_oldest_from_matrix()

        # Expected: full rebuild on the current deque.
        if len(s._events) == 0:
            assert s._history_matrix is None
            continue
        expected = generate_allen_matrix_from_history(list(s._events)).matrix
        actual = s._history_matrix
        assert actual is not None, f"step {i}: matrix is None"
        assert actual.shape == expected.shape, (
            f"step {i}: shape mismatch {actual.shape} vs {expected.shape}"
        )
        # Element-wise compare. dtype=object so use Python ==.
        for r in range(actual.shape[0]):
            for c in range(actual.shape[1]):
                assert actual[r, c] == expected[r, c], (
                    f"step {i}: matrix[{r},{c}] mismatch: "
                    f"actual={actual[r,c]!r} expected={expected[r,c]!r}\n"
                    f"actual:\n{actual}\nexpected:\n{expected}"
                )


def test_match_rule_to_matrix_equivalence_with_match_rule_to_history():
    """For a representative set of (rule, history) pairs, the two match
    functions return the same boolean."""
    from alpha_rule.wrappers.history_to_rule import HistoryToRuleWrapperBase

    rules = [
        "A",
        "A B <",
        "A B < A < <",
    ]
    histories = [
        [_make_event("A", 0, 2)],
        [_make_event("A", 0, 2), _make_event("B", 3, 5)],
        [
            _make_event("A", 0, 2),
            _make_event("B", 3, 5),
            _make_event("A", 6, 8),
        ],
        [
            _make_event("C", 0, 1),
            _make_event("A", 2, 4),
            _make_event("B", 5, 7),
            _make_event("A", 8, 10),
        ],
    ]

    for rule_str in rules:
        rule = AllenMatrix.from_hierarchy_string(rule_str)
        for history in histories:
            # Build incremental matrix by feeding events one at a time.
            class _StandIn:
                _append_event_to_matrix = HistoryToRuleWrapperBase._append_event_to_matrix
                _trim_oldest_from_matrix = HistoryToRuleWrapperBase._trim_oldest_from_matrix

                def __init__(self):
                    from collections import deque
                    self._events = deque()
                    self._history_matrix = None

            s = _StandIn()
            for ev in history:
                s._append_event_to_matrix(ev)
                s._events.append(ev)

            via_history = match_rule_to_history(rule, history)
            incremental = match_rule_to_matrix(rule, s._history_matrix)
            assert via_history == incremental, (
                f"rule={rule_str!r}, history={history!r}: "
                f"via_history={via_history} incremental={incremental}"
            )


def test_match_rule_to_matrix_with_none_history_returns_false():
    rule = AllenMatrix.from_hierarchy_string("A")
    assert match_rule_to_matrix(rule, None) is False


def test_trim_drops_rightmost_column():
    """After enough events to overflow the window, the incremental
    matrix should match the full rebuild on the trimmed deque."""
    from alpha_rule.wrappers.history_to_rule import HistoryToRuleWrapperBase

    class _StandIn:
        _append_event_to_matrix = HistoryToRuleWrapperBase._append_event_to_matrix
        _trim_oldest_from_matrix = HistoryToRuleWrapperBase._trim_oldest_from_matrix

        def __init__(self):
            from collections import deque
            self._events = deque()
            self._history_matrix = None
            self.window = 3

    events = [
        _make_event("A", 0, 1),
        _make_event("B", 2, 3),
        _make_event("C", 4, 5),
        _make_event("A", 6, 7),
        _make_event("B", 8, 9),
    ]

    s = _StandIn()
    for ev in events:
        s._append_event_to_matrix(ev)
        s._events.append(ev)
        while len(s._events) >= s.window:
            s._events.popleft()
            s._trim_oldest_from_matrix()

    # The deque should now hold the last few events, trimmed to <= window-1.
    expected = generate_allen_matrix_from_history(list(s._events)).matrix
    actual = s._history_matrix
    assert actual.shape == expected.shape
    for r in range(actual.shape[0]):
        for c in range(actual.shape[1]):
            assert actual[r, c] == expected[r, c], (
                f"[{r},{c}] actual={actual[r,c]!r} expected={expected[r,c]!r}"
            )
