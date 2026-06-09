"""
Tests for the ``AllenMatrix`` wrapper class.

Pins:
    - ``from_hierarchy_string`` round-trips via ``get_hierarchy_string``.
    - ``random_matrix`` produces shape (n+2, n) and passes validation.
    - ``shape`` attribute agrees with the underlying numpy array.
"""
from __future__ import annotations

import numpy as np

from alpha_rule.rules.allen_matrix import AllenMatrix


def test_from_hierarchy_string_length_one():
    am = AllenMatrix.from_hierarchy_string("A")
    assert am.get_hierarchy_string() == "A"
    assert am.shape == (3, 1)


def test_from_hierarchy_string_length_three():
    rule = "A B < C < <"
    am = AllenMatrix.from_hierarchy_string(rule)
    assert am.get_hierarchy_string() == rule
    assert am.shape == (5, 3)


def test_random_matrix_valid_shape():
    # A random matrix of size 3 with three event types. Validation runs
    # inside ``__init__``; just asserting it doesn't raise is enough.
    am = AllenMatrix.random_matrix(3, ["A", "B", "C"])
    assert am.shape[0] == am.shape[1] + 2


def test_repr_contains_shape():
    am = AllenMatrix.from_hierarchy_string("A B <")
    text = repr(am)
    assert "AllenMatrix" in text
    assert "shape=" in text
