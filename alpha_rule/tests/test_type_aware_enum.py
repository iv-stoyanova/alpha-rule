"""
Tests for the type-aware binary subset enumeration.

Pins:
    - ``enumerate_type_matched_vectors`` only yields vectors whose
      selected positions match the rule's event types in order: a strict
      subset of the unconstrained ``generate_binary_vectors_fixed_sum``
      output.
    - ``match_rule_to_history`` agrees with a brute-force reference that
      enumerates every candidate and checks ``matrix_left_match``.
    - Wildcard rule positions (``"#"`` in the rule's event-type row) are
      handled correctly.
"""
from __future__ import annotations

import itertools
import random

from alpha_rule.helpers.generic import Event
from alpha_rule.rules.allen_matrix import AllenMatrix
from alpha_rule.rules.rule_matching import (
    apply_binary_vector,
    enumerate_type_matched_vectors,
    generate_allen_matrix_from_history,
    generate_binary_vectors_fixed_sum,
    match_rule_to_history,
    match_rule_to_matrix,
    matrix_left_match,
)


def _ev(t, start, end):
    return Event(t, start, end)


def test_enumerate_type_matched_subset_of_unconstrained():
    """Every vector yielded by the type-aware enumerator must also be
    yielded by the unconstrained enumerator (which has no type
    constraint)."""
    history_types = ["A", "B", "A", "C", "A"]
    rule_types = ["A", "B", "A"]
    k = len(rule_types)
    n = len(history_types)

    unconstrained = set(tuple(v) for v in generate_binary_vectors_fixed_sum(n, k))
    typed = set(tuple(v) for v in enumerate_type_matched_vectors(history_types, rule_types))

    assert typed.issubset(unconstrained), f"{typed} not subset of {unconstrained}"
    # typed must be STRICTLY smaller here: B at position 1 rules out
    # positions whose type is A or C.
    assert len(typed) < len(unconstrained)


def test_enumerate_type_matched_respects_position_constraint():
    """Each yielded vector's selected positions must satisfy the type
    constraint at each rule position."""
    history_types = ["A", "B", "A", "C", "A"]
    rule_types = ["A", "C", "A"]

    for vec in enumerate_type_matched_vectors(history_types, rule_types):
        # Selected positions in order.
        picks = [i for i, b in enumerate(vec) if b == 1]
        assert len(picks) == len(rule_types)
        for r, p in enumerate(picks):
            assert history_types[p] == rule_types[r], (
                f"picks={picks}, rule_types={rule_types}, "
                f"history_types[{p}]={history_types[p]} != {rule_types[r]}"
            )


def test_enumerate_type_matched_wildcard_rule_position():
    """A ``"#"`` in the rule types acts as a wildcard, so any history type
    matches at that position."""
    history_types = ["A", "B", "C"]
    rule_types = ["A", "#"]

    vecs = list(enumerate_type_matched_vectors(history_types, rule_types))
    # Should yield TWO candidates: (pos0=A, pos1=B) and (pos0=A, pos2=C).
    selected = sorted(
        tuple(i for i, b in enumerate(v) if b == 1) for v in vecs
    )
    assert selected == [(0, 1), (0, 2)]


def test_enumerate_type_matched_empty_when_first_type_mismatch():
    """If the newest event type doesn't match the rule's first position,
    no candidates can match."""
    history_types = ["B", "A", "A"]
    rule_types = ["A", "A"]
    assert list(enumerate_type_matched_vectors(history_types, rule_types)) == []


def test_enumerate_type_matched_empty_when_too_few_types():
    """If the history has fewer matching positions than the rule needs,
    no candidates."""
    history_types = ["A", "B", "B"]
    rule_types = ["A", "B", "B", "B"]
    # n=3, k=4 means no candidates (k > n).
    assert list(enumerate_type_matched_vectors(history_types, rule_types)) == []


def test_match_rule_to_history_runs_on_representative_cases():
    """``match_rule_to_history`` runs without raising on a representative
    set of (rule, history) pairs: positive matches, type-mismatch
    rejections, length-too-short rejections, and multi-event rules."""
    cases = [
        ("A", [_ev("A", 0, 1)]),
        ("A", [_ev("B", 0, 1)]),
        ("B A <", [_ev("A", 0, 1), _ev("B", 5, 6)]),
        ("B A <", [_ev("A", 0, 1), _ev("A", 5, 6)]),
        ("A B < A < <", [_ev("A", 0, 1), _ev("B", 5, 6), _ev("A", 10, 11)]),
    ]
    for rule_str, history in cases:
        rule = AllenMatrix.from_hierarchy_string(rule_str)
        # The actual outcomes are checked in the cases below.
        match_rule_to_history(rule, history)


def test_match_rule_to_history_expected_outcomes():
    """``match_rule_to_history`` returns the expected boolean on a set of
    known (rule, history) cases."""
    cases = [
        # (rule_str, history, expected_match)
        ("A", [_ev("B", 0, 1), _ev("A", 2, 3)], True),
        ("A", [_ev("A", 0, 1), _ev("B", 2, 3)], False),
        ("A B <", [_ev("B", 0, 1)], False),  # too few events
        ("B A <", [_ev("A", 0, 1), _ev("B", 5, 6)], True),
        ("B A <", [_ev("A", 0, 1), _ev("A", 5, 6)], False),
    ]
    for rule_str, history, expected in cases:
        rule = AllenMatrix.from_hierarchy_string(rule_str)
        actual = match_rule_to_history(rule, history)
        assert actual is expected, (
            f"rule={rule_str!r} history={[(e.type, e.start, e.end) for e in history]} "
            f"expected={expected} got={actual}"
        )


def test_type_aware_matches_brute_force_reference():
    """For a set of (rule, history) pairs, the type-aware
    ``match_rule_to_history`` yields the same result as a brute-force
    reference that enumerates every size-n candidate (leftmost position
    forced) and checks ``matrix_left_match`` on each.
    """
    rule_strings = [
        "A",
        "B A <",
        "A B <",
    ]
    histories = [
        [_ev("A", 0, 1)],
        [_ev("A", 0, 1), _ev("B", 5, 6)],
        [_ev("B", 0, 1), _ev("A", 2, 3)],
        [_ev("A", 0, 1), _ev("A", 5, 6), _ev("B", 10, 11)],
        [_ev("A", 0, 1), _ev("B", 5, 6), _ev("A", 10, 11)],
    ]

    for rule_str in rule_strings:
        rule = AllenMatrix.from_hierarchy_string(rule_str)
        for history in histories:
            new_result = match_rule_to_history(rule, history)
            # Manual reference: build matrix, enumerate ALL candidates,
            # try each.
            n = rule.shape[1]
            if len(history) == 0:
                ref = False
            else:
                if history[-1].type != rule.matrix[1, 0]:
                    ref = False
                elif n == 1:
                    ref = True
                else:
                    allowed = set(rule.matrix[1])
                    filt = [e for e in history if "#" in allowed or e.type in allowed]
                    if len(filt) < n:
                        ref = False
                    else:
                        m = generate_allen_matrix_from_history(filt)
                        ref = False
                        # Brute-force: try ALL combinations of size n with leftmost forced.
                        positions = list(range(1, len(filt)))
                        for combo in itertools.combinations(positions, n - 1):
                            vec = [0] * len(filt)
                            vec[0] = 1
                            for p in combo:
                                vec[p] = 1
                            cand = apply_binary_vector(m, vec)
                            if matrix_left_match(cand, rule):
                                ref = True
                                break
            assert new_result is ref, (
                f"rule={rule_str!r} hist={[(e.type, e.start, e.end) for e in history]}: "
                f"new={new_result} ref={ref}"
            )


# --------------------------------------------------------------------------- #
# Fused forward-checking matcher: targeted pins + a broad brute-force fuzz.
# --------------------------------------------------------------------------- #

def test_fused_matcher_respects_relation_orientation():
    """The relation cell is read as history[picks[s]+2, pos] (upper
    triangle). Transposing it would read the "#" lower triangle and wrongly
    accept. Same types, opposite temporal order -> opposite result."""
    rule = AllenMatrix.from_hierarchy_string("A B <")  # newest A, earlier B, B before A
    assert match_rule_to_history(rule, [Event("B", 0, 1), Event("A", 5, 6)]) is True
    # Earlier B now comes AFTER A: the relation is ">", not "<".
    assert match_rule_to_history(rule, [Event("B", 5, 6), Event("A", 0, 1)]) is False


def test_fused_matcher_wildcard_type_matches_any():
    """A "#" in the rule's type row matches any history type at that
    position, end-to-end through the fused consumer (the enumerate tests
    only cover the enumerator)."""
    rule = AllenMatrix.from_hierarchy_string("A # <")  # newest A, earlier ANY, before A
    assert match_rule_to_history(rule, [Event("C", 0, 1), Event("A", 5, 6)]) is True
    assert match_rule_to_history(rule, [Event("B", 0, 1), Event("A", 5, 6)]) is True
    # The relation is still enforced even with a wildcard type.
    assert match_rule_to_history(rule, [Event("C", 5, 6), Event("A", 0, 1)]) is False


def test_fused_matcher_indicator_row_is_load_bearing():
    """The indicator row participates in the match. Position 0 (newest) is
    always picked, so corrupting the rule's column-0 indicator must flip a
    matching case to a non-match -- proving the check was not dropped."""
    history = [Event("A", 0, 1), Event("B", 2, 5), Event("A", 6, 7)]
    hm = generate_allen_matrix_from_history(history)
    rule = apply_binary_vector(hm, [1, 1, 1])  # the whole history as a rule
    assert match_rule_to_matrix(rule, hm.matrix) is True

    corrupted = AllenMatrix(rule.matrix.copy(), validate=False)
    corrupted.matrix[0, 0] = 1 - int(corrupted.matrix[0, 0])
    assert match_rule_to_matrix(corrupted, hm.matrix) is False


def test_fused_matcher_matches_brute_force_over_random_patterns():
    """Broad equivalence: random histories with extracted (sliced-indicator)
    rules, compared against the brute-force enumerate-all + matrix_left_match
    reference. Exercises types, Allen relations and the indicator row far
    beyond the canonical cases above."""
    rng = random.Random(7)
    types = ["A", "B", "C", "D"]

    def rand_hist(num):
        h, t = [], 0
        for _ in range(num):
            d = rng.randint(1, 6)
            s = t
            e = s + d
            h.append(Event(rng.choice(types), s, e))
            t = e + rng.randint(-d + 1, 2)
        return h

    def brute_ref(rule, history):
        n = rule.shape[1]
        if not history or history[-1].type != rule.matrix[1, 0]:
            return False
        if n == 1:
            return True
        allowed = set(rule.matrix[1])
        filt = [e for e in history if "#" in allowed or e.type in allowed]
        if len(filt) < n:
            return False
        m = generate_allen_matrix_from_history(filt)
        for combo in itertools.combinations(range(1, len(filt)), n - 1):
            vec = [0] * len(filt)
            vec[0] = 1
            for p in combo:
                vec[p] = 1
            if matrix_left_match(apply_binary_vector(m, vec), rule):
                return True
        return False

    for _ in range(600):
        src = rand_hist(rng.randint(2, 8))
        m = generate_allen_matrix_from_history(src)
        k = rng.randint(1, min(4, len(src)))
        idx = sorted(rng.sample(range(1, len(src)), k - 1)) if k > 1 else []
        vec = [1] + [1 if i in idx else 0 for i in range(len(src) - 1)]
        rule = apply_binary_vector(m, vec)
        hist = rand_hist(rng.randint(1, 9))
        got = match_rule_to_history(rule, hist)
        exp = brute_ref(rule, hist)
        assert got is exp, (
            rule.get_hierarchy_string(),
            [(e.type, e.start, e.end) for e in hist],
            got,
            exp,
        )
