import random
import numpy as np
from enum import Enum


class AllenRelation(Enum):
    """Defines Allen's Temporal Interval Relations"""
    BEFORE = "<"
    MEETS = "m"
    OVERLAPS = "o"
    STARTS = "s"
    DURING = "d"
    FINISHES = "f"
    EQUALS = "="
    AFTER = ">"
    MET_BY = "mi"
    OVERLAPPED_BY = "oi"
    STARTED_BY = "si"
    CONTAINS = "di"
    FINISHED_BY = "fi"

    @classmethod
    def all_relations(cls):
        return list(cls._value2member_map_.keys())


allen_relations = AllenRelation.all_relations()

# Every legal relation-matrix cell value: the 13 Allen symbols plus the "#"
# filler/wildcard. Built once at import rather than rebuilt on every
# ``validate_standard_matrix`` call.
VALID_RELATIONS = set(allen_relations) | {"#"}


def validate_standard_matrix(matrix):
    """
    Validates a standard Allen matrix.

    :param matrix: NumPy array representing the Allen matrix.
    :raises ValueError: If the matrix does not meet validation criteria.
    """
    if matrix is None:
        raise ValueError("Matrix is None. Provide a valid Allen matrix.")

    rows, cols = matrix.shape

    # The matrix must have (n+2) rows and n columns
    if rows != cols + 2:
        raise ValueError(f"Invalid matrix shape: expected (n+2, n), got {matrix.shape}")

    # Validate indicator row (only 0s and 1s)
    expected_indicator = check_rows_columns_combined(matrix[1:])  # Compute correct indicator row
    actual_indicator = matrix[0].astype(int)  # Convert indicator row to integer array

    if not np.array_equal(actual_indicator, expected_indicator):
        raise ValueError(
            f"Incorrect indicator row: expected {expected_indicator.tolist()}, got {actual_indicator.tolist()}")


    # Validate relation matrix (VALID_RELATIONS is built once at import)
    for i in range(2, rows):
        for j in range(cols):
            value = matrix[i, j]

            # Check if value is a valid Allen relation or "#"
            if value not in VALID_RELATIONS:
                raise ValueError(f"Invalid relation at ({i}, {j}): '{value}'")

            # Check diagonal elements are "="
            if i - 2 == j and value != "=":
                raise ValueError(f"Diagonal element at ({i}, {j}) must be '='.")

            # Ensure lower triangle (below diagonal) is "#"
            if i - 2 > j and value != "#":
                raise ValueError(f"Lower triangular element at ({i}, {j}) must be '#'.")


def check_rows_columns_combined(matrix, values_to_remove=['#', '=']):
    """
    Check which rows and columns in the matrix contain only the specified values.
    Return a single list where 1 means either the row or the column at that index
    contains only the target values and should be removed, and 0 means the row or
    column contains other values and should be kept.

    :param matrix: 2D NumPy array (matrix)
    :param values_to_remove: The list of values to check for (default is ['#', '='])
    :return: A single list of 0s and 1s indicating rows/columns to keep or remove.
    """
    # Create a mask that checks if each element in the matrix is in values_to_remove
    mask = np.isin(matrix, values_to_remove)

    # Check for rows: np.all checks if all elements in each row are in the values_to_remove
    rows_check = np.all(mask[1:, :], axis=1)

    # Check for columns: np.all checks if all elements in each column are in the values_to_remove
    columns_check = np.all(mask, axis=0)

    # Combine the results: we want to remove rows/columns where all elements are in values_to_remove
    combined_check = np.logical_not(np.logical_and(rows_check, columns_check))

    # Convert the boolean array to a list of 0s and 1s
    return combined_check


def generate_matrix(n, possible_event_types):
    """
    Generate a random (n+1) x n matrix, clean it, refill it, and add an indicator row.
    """
    matrix = np.empty((n + 1, n), dtype=object)
    matrix[0] = sample_with_geometric_distribution(possible_event_types, n, 0.7)

    # Fill the matrix with random Allen relations or '#'
    for i in range(0, n):
        for j in range(n):
            if i == j:
                matrix[i + 1, j] = '='  # Diagonal elements are '='
            elif j < i:
                matrix[i + 1, j] = '#'  # Below diagonal is empty
            else:
                if random.random() < 0.5:
                    matrix[i + 1, j] = random.choice(allen_relations)  # Random Allen relation
                else:
                    matrix[i + 1, j] = "#"

    final_matrix = regular_matrix(matrix, n)

    return final_matrix


def sample_with_geometric_distribution(items, n, p_empty=0.3):
    """
    Randomly samples `n` values from `items`, using a geometric distribution to determine '#' gaps.
    """
    sampled_list = []
    while len(sampled_list) < n:
        sampled_list.append("#") if np.random.random() < p_empty \
            else sampled_list.append(np.random.choice(items))
    return sampled_list[:n]


def regular_matrix(matrix, n):
    # Clean the matrix to remove unnecessary rows and columns
    cleaned_matrix = clean_empty_rows_columns(matrix)

    # Refill the matrix back to size n and get the size of the cleaned matrix
    refilled_matrix, current_size = refill_matrix(cleaned_matrix, n)

    # Add the indicator row at the top
    final_matrix = add_indicator_row(refilled_matrix, current_size, n)
    return final_matrix


def clean_empty_rows_columns(matrix):
    """
    Remove rows and columns from the matrix where all elements are either '#' or '='.

    :param matrix: 2D NumPy array (matrix)
    :return: A new matrix with empty rows and columns removed.
    """
    empty_mask = check_rows_columns_combined(matrix, values_to_remove=['#', '='])
    new_matrix = remove_rows_columns_based_on_mask(matrix, empty_mask)
    return new_matrix


def refill_matrix(matrix, n):
    """
    Refill the matrix after cleaning to the size n, adding '#' and '=' where necessary.
    """
    current_size = matrix.shape[1]
    filled_matrix = np.full((n + 1, n), "#", dtype=object)  # Create an empty matrix filled with '#'

    # Copy the cleaned matrix into the upper left part
    filled_matrix[:matrix.shape[0], :matrix.shape[1]] = matrix

    # Set the diagonal elements to '='
    for i in range(n):
        filled_matrix[i + 1, i] = '='

    return filled_matrix, current_size


def add_indicator_row(matrix, current_size, n):
    """
    Add an indicator row of 0s and 1s to mark which columns have been filled (1)
    and which were generated during refill (0).
    """
    # Indicator row: 0 for columns from the original cleaned matrix, 1 for new columns
    indicator_row = np.zeros(n, dtype=int)
    indicator_row[:current_size] = 1  # Mark columns from the original cleaned matrix as 0

    # Insert the indicator row at the top of the matrix
    matrix_with_indicator = np.vstack([indicator_row, matrix])
    return matrix_with_indicator


def remove_rows_columns_based_on_mask(matrix, keep):
    """
    Remove rows and columns from the matrix based on the binary mask.

    :param matrix: 2D NumPy array (matrix)
    :param mask: List or 1D NumPy array of 0s and 1s, where 1 means remove the corresponding row/column
    :return: A new matrix with specified rows and columns removed
    """

    extended = np.insert(keep, 0, True)

    # Use boolean indexing to filter rows and columns
    return matrix[extended][:, keep]


def hierarchy_string_to_matrix(hierarchy_string):
    """
    Converts a flat hierarchy string back into a valid Allen matrix.

    ``<END>`` is a valid marker on a rule name (meaning "the rule
    terminated here") but is NOT an Allen relation, so it is stripped
    before parsing. Callers can pass a raw rule name like
    ``"A B < <END>"`` without pre-processing.

    :param hierarchy_string: String representing the hierarchy.
    :return: NumPy matrix representing the Allen matrix.
    """
    elements = [t for t in hierarchy_string.split() if t != "<END>"]
    n = 0  # Number of events (determined while parsing)
    col_data = []  # Stores parsed columns

    while elements:
        column = []
        n += 1  # Increment event count
        for _ in range(n):
            if elements:
                column.append(elements.pop(0))
            else:
                column.append("#")  # Pad missing values with "#"
        col_data.append(column)

    # Pad all columns to the same length using "#" for consistency
    max_length = len(col_data[-1])
    for col in col_data:
        while len(col) < max_length:
            col.append("#")  # Pad with "#"

    # Convert parsed columns to row-major order (transposing back)
    col_data = np.array(col_data, dtype=object).T

    # Initialize (n+2) x n Allen matrix
    matrix = np.full((n + 2, n), "#", dtype=object)

    # Assign type filter row (1st row after indicator)
    matrix[1] = col_data[0]

    # Fill relations from parsed data
    matrix[2: (2 + (n - 1))] = col_data[1:, ]

    # Fill diagonal with "="
    for i in range(n):
        matrix[i + 2, i] = "="

    # Compute indicator row dynamically
    matrix[0] = check_rows_columns_combined(matrix[1:]).astype(int)

    # Validate and return the matrix
    validate_standard_matrix(matrix)
    return matrix


def matrix_to_hierarchy_string(matrix):
    """
    Converts an Allen matrix into a hierarchical string format.

    :param matrix: NumPy array representing the Allen matrix.
    :return: A formatted string representing the hierarchy.
    """
    rows, cols = matrix.shape

    if rows != cols + 2:
        raise ValueError(f"Invalid matrix shape: expected (n+2, n), got {matrix.shape}")

    hierarchy_list = []

    # Iterate over columns (Transpose the relation structure)
    for i, col in enumerate(matrix[1:, :].T):  # Transpose to get hierarchy order
        for j in range(i + 1):  # Only take elements up to the diagonal
            if j == 0:
                hierarchy_list.append(f"{col[j]}")  # Type format
            else:
                hierarchy_list.append(f"{col[j]}")  # Relation format

    return " ".join(hierarchy_list)
