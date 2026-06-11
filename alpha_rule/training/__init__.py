"""
Training package, outer self-play loop and orchestration.

The MVP entry point is ``train`` (see ``training.train``), which wires
``run_self_play`` + ``ReplayBuffer`` + ``train_step`` together.

``train``'s top-level imports are intentionally light (grammar, replay,
self-play); the heavy ones (``torch``, model classes) are lazy-imported
inside the function body so this package's import is fast.
"""
from alpha_rule.training.csv_logger import (  # noqa: F401
    AlphaZeroCSVLogger,
    CSV_COLUMNS,
    RUN_EVAL_CSV_COLUMNS,
    RunLevelEvalLogger,
    load_alphazero_logs,
    load_run_level_eval_logs,
)
from alpha_rule.training.train import (  # noqa: F401
    IterationLog,
    TrainingLog,
    play,
    play_top_k,
    train,
)

__all__ = [
    "AlphaZeroCSVLogger",
    "CSV_COLUMNS",
    "RUN_EVAL_CSV_COLUMNS",
    "IterationLog",
    "RunLevelEvalLogger",
    "TrainingLog",
    "load_alphazero_logs",
    "load_run_level_eval_logs",
    "play",
    "play_top_k",
    "train",
]
