"""
Tests for ``alpha_rule.rules.rule_matching``.

Pins:
    - ``determine_allen_relation`` produces the correct symbol for every
      canonical Allen relation, covering all 13 cases.
    - ``generate_binary_vectors_fixed_sum`` returns vectors that sum to k,
      always include index 0, and respect the prefix-extension invariant.
    - ``match_rule_to_history`` correctly handles the length-1 short-circuit,
      simple positive matches, and filters out histories too short to match.
"""
from __future__ import annotations

import random

import numpy as np

from alpha_rule.helpers.generic import Event
from alpha_rule.rules.allen_matrix import AllenMatrix
from alpha_rule.rules.rule_matching import (
    determine_allen_relation,
    generate_allen_matrix_from_history,
    generate_binary_vectors_fixed_sum,
    match_rule_to_history,
)


# --------------------------------------------------------------------------- #
# determine_allen_relation: exhaustive
# --------------------------------------------------------------------------- #

def _ev(a_start, a_end):
    return Event("A", a_start, a_end)


def test_determine_allen_relation_before():
    assert determine_allen_relation(_ev(0, 1), _ev(5, 6)) == "<"


def test_determine_allen_relation_after():
    assert determine_allen_relation(_ev(10, 11), _ev(0, 1)) == ">"


def test_determine_allen_relation_meets():
    assert determine_allen_relation(_ev(0, 5), _ev(5, 10)) == "m"


def test_determine_allen_relation_met_by():
    assert determine_allen_relation(_ev(5, 10), _ev(0, 5)) == "mi"


def test_determine_allen_relation_overlaps():
    assert determine_allen_relation(_ev(0, 5), _ev(3, 8)) == "o"


def test_determine_allen_relation_overlapped_by():
    assert determine_allen_relation(_ev(3, 8), _ev(0, 5)) == "oi"


def test_determine_allen_relation_starts():
    assert determine_allen_relation(_ev(0, 3), _ev(0, 8)) == "s"


def test_determine_allen_relation_started_by():
    assert determine_allen_relation(_ev(0, 8), _ev(0, 3)) == "si"


def test_determine_allen_relation_during():
    assert determine_allen_relation(_ev(2, 4), _ev(0, 8)) == "d"


def test_determine_allen_relation_contains():
    assert determine_allen_relation(_ev(0, 8), _ev(2, 4)) == "di"


def test_determine_allen_relation_finishes():
    assert determine_allen_relation(_ev(3, 8), _ev(0, 8)) == "f"


def test_determine_allen_relation_finished_by():
    assert determine_allen_relation(_ev(0, 8), _ev(3, 8)) == "fi"


def test_determine_allen_relation_equals():
    assert determine_allen_relation(_ev(0, 5), _ev(0, 5)) == "="


# --------------------------------------------------------------------------- #
# generate_binary_vectors_fixed_sum
# --------------------------------------------------------------------------- #

def test_binary_vectors_empty_when_k_too_large():
    assert generate_binary_vectors_fixed_sum(3, 5) == []


def test_binary_vectors_k_equals_n_yields_all_ones():
    vectors = generate_binary_vectors_fixed_sum(3, 3)
    assert vectors == [[1, 1, 1]]


def test_binary_vectors_always_include_position_zero():
    vectors = generate_binary_vectors_fixed_sum(4, 2)
    assert all(v[0] == 1 for v in vectors)
    assert all(sum(v) == 2 for v in vectors)


def test_binary_vectors_prefix_extension_keeps_prefix_ones():
    prefix = [[1, 1, 0, 0, 0]]
    extensions = generate_binary_vectors_fixed_sum(5, 3, prefix_vectors=prefix)
    # Each extension must have the original two 1s still set.
    for vec in extensions:
        assert vec[0] == 1
        assert vec[1] == 1
        assert sum(vec) == 3


# --------------------------------------------------------------------------- #
# match_rule_to_history
# --------------------------------------------------------------------------- #

def test_match_length_one_rule_against_matching_last_event():
    rule = AllenMatrix.from_hierarchy_string("A")
    history = [Event("B", 0, 1), Event("A", 2, 3)]
    assert match_rule_to_history(rule, history) is True


def test_match_length_one_rule_rejects_wrong_last_event():
    rule = AllenMatrix.from_hierarchy_string("A")
    history = [Event("A", 0, 1), Event("B", 2, 3)]
    assert match_rule_to_history(rule, history) is False


def test_match_rejects_history_shorter_than_rule():
    rule = AllenMatrix.from_hierarchy_string("A B <")
    # last event matches type but there are not enough filtered events.
    history = [Event("B", 0, 1)]
    assert match_rule_to_history(rule, history) is False


def test_match_two_event_rule_positive():
    # The matrix stores history reversed, so matrix[1, 0] is the LAST event.
    # The pattern "earlier A, later B in history" serialises to "B A <":
    # B is the last event, A is earlier, and the relation from earlier to
    # later is "<" (before).
    rule = AllenMatrix.from_hierarchy_string("B A <")
    history = [Event("A", 0, 1), Event("B", 5, 6)]
    assert match_rule_to_history(rule, history) is True


def test_match_two_event_rule_last_event_type_mismatch():
    rule = AllenMatrix.from_hierarchy_string("B A <")
    history = [Event("A", 0, 1), Event("A", 5, 6)]
    # Last event isn't type B, so the short-circuit returns False.
    assert match_rule_to_history(rule, history) is False


# --------------------------------------------------------------------------- #
# generate_allen_matrix_from_history: the vectorised builder must reproduce
# the scalar determine_allen_relation double-loop exactly.
# --------------------------------------------------------------------------- #

def _scalar_rel_rows(history):
    """Reference: the original scalar build of rows[1:] (types + relations)."""
    n = len(history)
    rows = np.full((n + 1, n), "#", dtype=object)  # row 0 = types, rows 1.. = relations
    rows[0] = [e.type for e in reversed(history)]
    for i in range(n):
        rows[i + 1, i] = "="
    for i in range(n):
        for j in range(i + 1, n):
            rel = determine_allen_relation(history[n - 1 - j], history[n - 1 - i])
            if rel:
                rows[i + 1, j] = rel
    return rows


def test_builder_matrix_is_object_dtype():
    m = generate_allen_matrix_from_history([Event("A", 0, 1), Event("B", 2, 3)])
    assert m.matrix.dtype == object


def test_builder_zero_length_coincident_is_meets_not_equals():
    # determine_allen_relation checks "m" (end == start) BEFORE "=" (equal
    # spans), so two coincident zero-length intervals classify as "m". The
    # vectorised np.select must preserve that ordering.
    m = generate_allen_matrix_from_history([Event("A", 0, 0), Event("B", 0, 0)])
    # cell [2, 1] is the relation between the two events.
    assert m.matrix[2, 1] == "m"


def test_builder_matches_scalar_reference_over_random_histories():
    rng = random.Random(2024)
    types = ["A", "B", "C", "D"]
    for _ in range(300):
        n = rng.randint(1, 10)
        history, t = [], 0
        for _ in range(n):
            d = rng.randint(0, 6)          # 0 exercises zero-length intervals
            s = t
            e = s + d
            history.append(Event(rng.choice(types), s, e))
            t = e + rng.randint(-d, 2)     # overlaps / negative gaps too
        built = generate_allen_matrix_from_history(history).matrix
        ref = _scalar_rel_rows(history)
        assert np.array_equal(built[1:].astype(str), ref.astype(str)), history
