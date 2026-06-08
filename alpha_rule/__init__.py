"""alpha-rule: AlphaZero style MCTS over Allen interval temporal rules.

The package is assembled one component at a time. Heavy subpackages (nn,
evaluation, analysis) import torch / plotly / scipy lazily, so importing
``alpha_rule`` on its own stays cheap and free of those dependencies.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
