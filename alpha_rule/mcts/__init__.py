"""
MCTS package, the AlphaZero self-play search over grammar productions.

    - ``node``      : MCTSRuleNode, pure state (stats + tree links + flags).
    - ``expansion`` : RuleExpansion, grammar-driven child-creation seam.
    - ``selection`` : SelectionStrategy + PUCTSelection.
    - ``backprop``  : BackpropStrategy + MaxRewardBackup + PercentileRewardBackup.
    - ``replay``    : Trajectory + ReplayBuffer (per-step root-value targets).
    - ``self_play`` : run_self_play, one self-play episode.
"""
from alpha_rule.mcts.backprop import (  # noqa: F401
    BackpropStrategy,
    MaxRewardBackup,
    PercentileRewardBackup,
)
from alpha_rule.mcts.expansion import ExpansionStrategy, RuleExpansion  # noqa: F401
from alpha_rule.mcts.node import MCTSRuleNode  # noqa: F401
from alpha_rule.mcts.replay import (  # noqa: F401
    DEFAULT_REWARD_FLOOR,
    ReplayBuffer,
    Trajectory,
    TrajectoryStep,
)
from alpha_rule.mcts.selection import PUCTSelection, SelectionStrategy  # noqa: F401
from alpha_rule.mcts.self_play import run_self_play  # noqa: F401

__all__ = [
    "BackpropStrategy",
    "DEFAULT_REWARD_FLOOR",
    "ExpansionStrategy",
    "MCTSRuleNode",
    "MaxRewardBackup",
    "PUCTSelection",
    "PercentileRewardBackup",
    "ReplayBuffer",
    "RuleExpansion",
    "SelectionStrategy",
    "Trajectory",
    "TrajectoryStep",
    "run_self_play",
]
