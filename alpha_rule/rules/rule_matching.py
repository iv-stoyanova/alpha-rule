import itertools
from typing import List, Optional

import numpy as np

from alpha_rule.helpers.generic import Event
from alpha_rule.helpers.matrix_operations import check_rows_columns_combined
from alpha_rule.rules.allen_matrix import AllenMatrix


def generate_binary_vectors_fixed_sum(n, k, prefix_vectors=None):
    """
    Generates binary vectors of length `n` with exactly `k` ones,
    ensuring the first event (leftmost) is always kept.
    Extends prior matched vectors by adding 1s to the right of max used index.

    :param n: Total length of the binary vector.
    :param k: Number of 1s in the binary vector (subset size).
    :param prefix_vectors: Optional list of previously matched vectors to extend.
    :return: List of valid binary vectors.
    """
    if k < 1 or n < k:
        return []

    result = []
    positions = list(range(1, n))  # leftmost (index 0) always kept
    required_ones = k - 1

    if prefix_vectors is None:
        for ones_pos in itertools.combinations(positions, required_ones):
            vec = [0] * n
            vec[0] = 1
            for pos in ones_pos:
                vec[pos] = 1
            result.append(vec)
    else:
        for prefix in prefix_vectors:
            prefix_indices = [i for i, bit in enumerate(prefix[1:], start=1) if bit == 1]
            max_index = max(prefix_indices) if prefix_indices else 0
            for i in range(max_index + 1, n):
                if prefix[i] == 0:
                    new_vec = prefix[:]
                    new_vec[i] = 1
                    if sum(new_vec) == k:
                        result.append(new_vec)

    return result


def enumerate_type_matched_vectors(history_types, rule_types):
    """
    Type-aware analogue of ``generate_binary_vectors_fixed_sum``.

    Yields binary vectors of length ``n = len(history_types)`` with
    exactly ``k = len(rule_types)`` ones, position ``0`` always set,
    such that for each one-position ``p_r`` (in increasing order),
    ``history_types[p_r]`` matches ``rule_types[r]`` (a ``"#"`` in
    the rule types acts as a wildcard).

    The type constraint is folded into the enumeration: only positions
    whose history type matches the corresponding rule position are
    considered, which keeps the candidate count small. Allen relations
    are still validated on each candidate by ``matrix_left_match``.
    """
    n = len(history_types)
    k = len(rule_types)
    if k < 1 or n < k:
        return

    # Position 0 (newest) is always part of every candidate. Caller
    # paths already pre-check ``history_types[0] == rule_types[0]``
    # (the "last event must match" short-circuit) but defend here.
    if not (rule_types[0] == "#" or history_types[0] == rule_types[0]):
        return

    if k == 1:
        vec = [0] * n
        vec[0] = 1
        yield vec
        return

    def walk(r, last_pos, picks):
        if r == k:
            vec = [0] * n
            for p in picks:
                vec[p] = 1
            yield vec
            return
        target = rule_types[r]
        for pos in range(last_pos + 1, n):
            if target == "#" or history_types[pos] == target:
                picks.append(pos)
                yield from walk(r + 1, pos, picks)
                picks.pop()

    yield from walk(1, 0, [0])


def _match_via_type_aware_enum(filtered_am, rule_matrix):
    """Type-aware enumeration + ``matrix_left_match``. Shared inner loop
    used by both ``match_rule_to_history`` and ``match_rule_to_matrix``."""
    history_types = list(filtered_am.matrix[1])
    rule_types = list(rule_matrix.matrix[1])
    for binary_vector in enumerate_type_matched_vectors(history_types, rule_types):
        candidate_matrix = apply_binary_vector(filtered_am, binary_vector)
        if matrix_left_match(candidate_matrix, rule_matrix):
            return True
    return False


def filter_event_type(event, etype):
    return etype == '#' or event.type == etype


def make_event_filter_from_matrix(rule_matrix: AllenMatrix):
    allowed_types = set(rule_matrix.matrix[1])
    return lambda event: any(filter_event_type(event, etype) for etype in allowed_types)


def matrix_left_match(candidate: AllenMatrix, rule: AllenMatrix) -> bool:
    """
    Compares the candidate matrix to the rule matrix from the left side.
    Each cell matches if:
    - Values are equal, or
    - Rule value is "#"

    :param candidate: AllenMatrix to validate.
    :param rule: AllenMatrix that defines the rule.
    :return: True if candidate matches rule (left-aligned), else False.
    """
    c_rows, c_cols = candidate.matrix.shape
    r_rows, r_cols = rule.matrix.shape

    if c_rows > r_rows or c_cols > r_cols:
        return False  # Candidate cannot be larger than rule

    for i in range(c_rows):
        for j in range(c_cols):
            rule_val = rule.matrix[i, j]
            cand_val = candidate.matrix[i, j]
            if rule_val != "#" and rule_val != cand_val:
                return False

    return True


def match_rule_to_matrix(
        rule_matrix: AllenMatrix,
        history_matrix: Optional[np.ndarray],
) -> bool:
    """
    Variant of :func:`match_rule_to_history` that takes a pre-built
    history matrix instead of rebuilding it from the events list.

    The wrapper (``HistoryToRuleWrapperBase``) maintains the history
    matrix incrementally, appending one column per env step rather than
    recomputing O(window²) Allen relations each step. This function then
    consumes the pre-built matrix and skips ``generate_allen_matrix_from_history``.

    Matrix layout (matching ``generate_allen_matrix_from_history``):
        shape ``(n+2, n)`` with column ``0`` = newest event.
        Row 0 = indicator, row 1 = types, rows 2..n+1 = Allen relations.

    Returns ``False`` when ``history_matrix`` is None or has zero
    columns, meaning the wrapper has not yet seen any events.
    """
    if history_matrix is None:
        return False

    n = rule_matrix.shape[1]
    n_history = history_matrix.shape[1]
    if n_history == 0:
        return False

    # Newest event must match the rule's last event type. The matrix
    # stores newest-first, so col 0 is the most recent event.
    last_event_type = history_matrix[1, 0]
    rule_last_event_type = rule_matrix.matrix[1, 0]
    if last_event_type != rule_last_event_type:
        return False
    elif n == 1:
        return True

    # Type filter: keep matrix columns whose event type is allowed by the
    # rule. A "#" in the rule's allowed set acts as a wildcard.
    allowed_types = set(rule_matrix.matrix[1])
    contains_wildcard = "#" in allowed_types
    keep_indices = [
        j for j, t in enumerate(history_matrix[1])
        if contains_wildcard or t in allowed_types
    ]
    length = len(keep_indices)
    if length < n:
        return False

    # Slice the matrix to the kept columns + corresponding relation rows.
    # Same pattern as ``apply_binary_vector``.
    filtered = history_matrix[:, keep_indices]
    filtered = filtered[[0, 1] + [i + 2 for i in keep_indices], :]
    filtered_am = AllenMatrix(filtered, validate=False)
    return _match_via_type_aware_enum(filtered_am, rule_matrix)


def match_rule_to_history(
        rule_matrix: AllenMatrix,
        history: List[Event]
) -> bool:
    """
    Matches a rule represented by an AllenMatrix to a filtered event history.

    :param rule_matrix: The AllenMatrix representing the rule to be matched.
    :param history: List of Event objects (most recent last).
    :param event_filter: Function to filter relevant events from history.
    :param match_func: Function that checks if a reduced AllenMatrix matches the rule.
    :return: True if a match is found, False otherwise.
    """

    n = rule_matrix.shape[1]

    # Always check last event matches the rule's last type
    last_event = history[-1]
    last_event_type = last_event.type
    rule_last_event_type = rule_matrix.matrix[1, 0]
    if last_event_type != rule_last_event_type:
        return False
    elif n == 1:
        return True

    event_filter = make_event_filter_from_matrix(rule_matrix)

    # Step 1: Filter the event history
    filtered_events = [e for e in history if event_filter(e)]
    length = len(filtered_events)

    if len(filtered_events) < n:
        return False  # Not enough events to match the rule

    # Step 2: Build Allen matrix from filtered events
    history_matrix = generate_allen_matrix_from_history(filtered_events)

    # Step 3: type-aware enumeration + matrix_left_match (shared with
    # match_rule_to_matrix).
    return _match_via_type_aware_enum(history_matrix, rule_matrix)


def generate_allen_matrix_from_history(history):
    """
    Generates an Allen matrix from a given history of events.

    :param history: List of Event objects.
    :return: AllenMatrix object representing the extracted rule.
    """
    if not history:
        raise ValueError("History cannot be empty.")

    n = len(history)  # Number of events in history
    matrix_size = n + 2  # Matrix includes indicator row and type row

    # Initialize matrix with "#"
    matrix = np.full((matrix_size, n), "#", dtype=object)

    # Step 1: Assign the event types (history is reversed)
    matrix[1] = [event.type for event in reversed(history)]

    # Step 2: Fill the diagonal with "="
    for i in range(n):
        matrix[i + 2, i] = "="  # Self-equality

    # Step 3: Compute Allen relations
    for i in range(n):
        for j in range(i + 1, n):  # Only fill upper triangle
            relation = determine_allen_relation(history[n - 1 - j], history[n - 1 - i])  # Reverse indexing
            if relation:
                matrix[i + 2, j] = relation  # Assign relation

    # Step 4: Compute indicator row dynamically
    matrix[0] = check_rows_columns_combined(matrix[1:]).astype(int)

    return AllenMatrix(matrix)


def determine_allen_relation(event1, event2):
    """
    Return the Allen interval relation between ``event1`` and ``event2``.

    Short-circuited: returns the first matching relation without evaluating
    the remaining conditions.
    """
    a, b = event1, event2
    if a.end < b.start:
        return "<"
    if a.end == b.start:
        return "m"
    if a.start < b.start < a.end < b.end:
        return "o"
    if a.start == b.start and a.end < b.end:
        return "s"
    if a.start > b.start and a.end < b.end:
        return "d"
    if a.start > b.start and a.end == b.end:
        return "f"
    if a.start == b.start and a.end == b.end:
        return "="
    if a.start > b.end:
        return ">"
    if a.start == b.end:
        return "mi"
    if b.start < a.start < b.end < a.end:
        return "oi"
    if a.start == b.start and a.end > b.end:
        return "si"
    if a.start < b.start and a.end > b.end:
        return "di"
    if a.start < b.start and a.end == b.end:
        return "fi"
    return None


def apply_binary_vector(matrix, binary_vector):
    """
    Applies a binary vector to an Allen matrix, keeping only selected events.

    The filtered submatrix is a strict slice of an already-validated parent,
    so we pass ``validate=False`` to ``AllenMatrix`` to skip the O(n²) re-scan
    in the inner rule-matching loop.

    :param matrix: AllenMatrix object.
    :param binary_vector: List of 0s and 1s indicating which events to keep.
    :return: Filtered AllenMatrix object.
    """

    keep_indices = [i for i, bit in enumerate(binary_vector) if bit == 1]

    filtered_matrix = matrix.matrix[:, keep_indices]
    filtered_matrix = filtered_matrix[[0, 1] + [i + 2 for i in keep_indices], :]

    return AllenMatrix(filtered_matrix, validate=False)
