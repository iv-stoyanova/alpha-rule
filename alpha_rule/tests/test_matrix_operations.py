"""
Tests for ``alpha_rule.helpers.matrix_operations``.

Pins:
    - The (n+2, n) shape contract enforced by ``validate_standard_matrix``.
    - The round-trip invariant
      ``matrix_to_hierarchy_string(hierarchy_string_to_matrix(s)) == s``
      for rules of varying lengths.
    - ``check_rows_columns_combined`` agrees with the hand-worked expectation
      on a matrix that contains a mix of wildcards and concrete relations.
"""
from __future__ import annotations

import numpy as np
import pytest

from alpha_rule.helpers.matrix_operations import (
    AllenRelation,
    check_rows_columns_combined,
    hierarchy_string_to_matrix,
    matrix_to_hierarchy_string,
    validate_standard_matrix,
)


# --------------------------------------------------------------------------- #
# validate_standard_matrix
# --------------------------------------------------------------------------- #

def test_validate_accepts_correct_length_one_matrix():
    # A length-1 rule: n=1 gives shape (3, 1). Indicator row, type row, and one
    # diagonal cell.
    matrix = np.array([[1], ["A"], ["="]], dtype=object)
    validate_standard_matrix(matrix)  # must not raise


def test_validate_rejects_wrong_shape():
    # (2, 1) is invalid. Expected (n+2, n) == (3, 1).
    matrix = np.array([["A"], ["="]], dtype=object)
    with pytest.raises(ValueError, match="Invalid matrix shape"):
        validate_standard_matrix(matrix)


def test_validate_rejects_bad_diagonal():
    matrix = np.array([[1, 1], ["A", "B"], ["<", "#"], ["#", "#"]], dtype=object)
    with pytest.raises(ValueError, match="Diagonal element"):
        validate_standard_matrix(matrix)


def test_validate_rejects_unknown_relation():
    matrix = np.array([[1, 1], ["A", "B"], ["=", "ZZ"], ["#", "="]], dtype=object)
    with pytest.raises(ValueError, match="Invalid relation"):
        validate_standard_matrix(matrix)


# --------------------------------------------------------------------------- #
# hierarchy_string_to_matrix / matrix_to_hierarchy_string round-trip
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "rule_string",
    [
        "A",
        "A B <",
        "A B < C < <",
        "A A =",          # same event type twice
    ],
)
def test_hierarchy_string_round_trip(rule_string):
    matrix = hierarchy_string_to_matrix(rule_string)
    recovered = matrix_to_hierarchy_string(matrix)
    assert recovered == rule_string


@pytest.mark.parametrize(
    "rule_string,canonical",
    [
        ("A <END>",            "A"),           # trailing terminal marker
        ("A B < <END>",        "A B <"),       # marker after a valid rule
        ("<END> A B <",        "A B <"),       # marker at the start (defensive)
        ("A <END> B <",        "A B <"),       # marker in the middle
        ("A B < <END> <END>",  "A B <"),       # multiple markers
    ],
)
def test_hierarchy_string_strips_end_marker_before_parsing(rule_string, canonical):
    """
    ``<END>`` is a legitimate marker on MCTS node names (means "rule
    terminated here") but NOT an Allen relation. The parser strips it
    before building the matrix so callers can pass a raw node.name
    without pre-processing. Round-tripping through the matrix recovers
    the canonical form without the marker.
    """
    matrix = hierarchy_string_to_matrix(rule_string)
    recovered = matrix_to_hierarchy_string(matrix)
    assert recovered == canonical


def test_hierarchy_string_to_matrix_shape():
    matrix = hierarchy_string_to_matrix("A B < C < <")
    assert matrix.shape == (5, 3)           # (n+2, n) with n=3


# --------------------------------------------------------------------------- #
# check_rows_columns_combined
# --------------------------------------------------------------------------- #

def test_check_rows_columns_combined_all_wildcards_marked_removable():
    # Two rows of relations; second column is all wildcards, so indicator = 0.
    matrix = np.array([
        ["A", "#"],
        ["=", "#"],
        ["#", "="],
    ], dtype=object)
    result = check_rows_columns_combined(matrix)
    # The second column is entirely '#'/'=' on rows [1:], so it should be
    # marked "removable" (combined_check = False = 0).
    assert list(result.astype(int)) == [1, 0]


# --------------------------------------------------------------------------- #
# AllenRelation enum sanity check
# --------------------------------------------------------------------------- #

def test_all_allen_relations_are_length_one_or_two():
    for value in AllenRelation.all_relations():
        assert 1 <= len(value) <= 2
        assert value in {
            "<", ">", "m", "mi", "o", "oi", "s", "si",
            "d", "di", "f", "fi", "=",
        }
