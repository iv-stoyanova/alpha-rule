"""
Tests for ``alpha_rule.evaluation.RuleStringNode``.

Pins the tiny duck-typed helper's guarantees:
    - frozen dataclass with a single ``.name`` string field
    - usable as an argument to any ``Evaluator.evaluate`` (just reads
      ``node.name``)
    - hashable (so it can live in sets/dicts if needed)
"""
from __future__ import annotations

import pytest

from alpha_rule.evaluation import RuleStringNode
from alpha_rule.evaluation.evaluator import EvalResult


class _NameReader:
    """Evaluator that just echoes node.name as the value's string."""
    def __init__(self):
        self.seen = []
    def evaluate(self, node):
        self.seen.append(node.name)
        return EvalResult(value=float(len(node.name)))  # dummy numeric


def test_rule_string_node_has_name_attribute():
    n = RuleStringNode(name="A B <")
    assert n.name == "A B <"


def test_rule_string_node_is_frozen():
    n = RuleStringNode(name="A")
    with pytest.raises(Exception):
        # frozen dataclass should reject mutation
        n.name = "B"


def test_rule_string_node_is_hashable():
    a = RuleStringNode(name="A B <")
    b = RuleStringNode(name="A B <")
    c = RuleStringNode(name="C")
    assert hash(a) == hash(b)
    assert {a, b, c} == {a, c}


def test_rule_string_node_works_with_evaluator():
    """``RuleStringNode`` is a valid argument to any ``Evaluator`` — the
    evaluator reads only ``node.name``."""
    inner = _NameReader()

    r1 = inner.evaluate(RuleStringNode(name="A B <"))
    r2 = inner.evaluate(RuleStringNode(name="different"))

    assert inner.seen == ["A B <", "different"]
    assert r1.value == float(len("A B <"))
    assert r2.value == float(len("different"))
