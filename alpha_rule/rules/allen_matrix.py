from alpha_rule.helpers.matrix_operations import (validate_standard_matrix,
                                                  generate_matrix, hierarchy_string_to_matrix,
                                                  matrix_to_hierarchy_string)


class AllenMatrix:
    """Represents a matrix of temporal interval relations"""

    def __init__(self, matrix=None, *, validate: bool = True):
        """
        Initialise an Allen matrix.

        Args:
            matrix: the (n+2, n) numpy array of types + relations.
            validate: run ``validate_standard_matrix`` (default). Pass
                False when you know the matrix is valid (e.g. filtering a
                submatrix out of one that was already validated) to avoid
                the O(n²) re-scan in the hot rule-matching loop.
        """
        self.matrix = matrix
        self.shape = matrix.shape

        if validate:
            validate_standard_matrix(self.matrix)

    @classmethod
    def random_matrix(cls, n, possible_event_types):
        """
        Generates a random Allen matrix and returns an instance of AllenMatrix.

        :param n: Size of the matrix (number of intervals).
        :param possible_event_types: List of event types.
        :return: An instance of AllenMatrix with a generated matrix.
        """
        matrix = generate_matrix(n, possible_event_types)
        return cls(matrix)

    @classmethod
    def from_hierarchy_string(cls, hierarchy_string):
        """
        Creates an AllenMatrix instance from a hierarchy string.

        :param hierarchy_string: A string describing the hierarchy.
        :return: An instance of AllenMatrix.
        """
        # ``hierarchy_string_to_matrix`` is the validating owner -- it already
        # runs ``validate_standard_matrix`` before returning, so skip the
        # redundant re-scan in ``__init__``.
        matrix = hierarchy_string_to_matrix(hierarchy_string)
        return cls(matrix, validate=False)

    def get_hierarchy_string(self):
        """Returns a hierarchical string representation of the matrix."""
        return matrix_to_hierarchy_string(self.matrix)

    def display(self):
        """Prints the matrix in a readable format."""
        print(self.matrix)

    def __repr__(self):
        """Returns a string representation of the Allen matrix."""
        return f"AllenMatrix(shape={self.shape})\n{self.matrix.__str__()}"