"""
Replay buffer + value-target computation for AlphaZero-style training.

Data flow:

    run_self_play() -> Trajectory(steps, value_scale)
                          |
        value_targets:  z_t = clip(state_value_t / value_scale, -1, +1)
                          |
    ReplayBuffer.push_trajectory(traj) writes (state, pi_t, z_t[, actions]) rows
                          |
              buffer.sample(batch_size) -> NN training step

Value target (z_t)
------------------
Each step carries ``state_value`` -- the per-step value the configured
``ValueTarget`` strategy read off the MCTS tree at that state (e.g.
``node.Q_max`` for the default ``MaxValue``; see ``mcts.value_target``). The
buffer maps it into the tanh value head's ``[-1, +1]`` range with a single
linear clip:

    z = clip(value / value_scale, -1, +1)

``value_scale`` is the simulator's positive reward cap (small, e.g. 1-3, read
off ``simulator.reward_scale``). Good rules then spread across ``[0, 1]``;
the long negative tail and ``-inf`` structural failures collapse onto ``-1``
(we only care that a bad rule is bad, not how bad). This replaces an earlier
symmetric ``[-100, +100]`` / 100 scheme that crushed the small real rewards
toward zero. A smoother alternative is ``z = tanh(value / value_scale)``,
which also sends ``-inf -> -1``; left as a future option.

Failures (``-inf``)
-------------------
``RuleSimulator.evaluate`` returns ``-inf`` for a rule that never matches (a
structural failure, not a low score). The clip sends it to ``-1`` so it
becomes a strong negative training signal rather than blowing up the loss; no
rows are dropped.
"""
from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class TrajectoryStep:
    """
    One MCTS root-step inside a self-play episode.

    Field grouping (the two halves are offset by one step -- do NOT combine
    ``reward`` and ``state_value`` as if they describe the same state):

        ``state`` / ``visit_pi`` / ``applicable_actions`` / ``state_value``
            all describe s_t, the state MCTS was rooted at.
        ``reward`` / ``next_state`` describe s_{t+1}, the child advanced to.
    """

    state: object
    """The state the policy ``visit_pi`` was computed at (an ``MCTSRuleNode``
    name string)."""

    visit_pi: Dict[str, float]
    """Production-name -> normalised MCTS visit probability at ``state``."""

    reward: float
    """R(next_state), reward of the child advanced to after sampling from
    ``visit_pi``. May be ``-inf`` for a failed evaluation."""

    next_state: Optional[object] = None
    """Name of the child that earned ``reward`` (s_{t+1})."""

    applicable_actions: Tuple[str, ...] = ()
    """Production names legal at ``state`` per the grammar; the training-time
    softmax mask source. Empty tuple = no mask."""

    state_value: Optional[float] = None
    """Value target z_t at ``state``, produced by the configured
    ``ValueTarget`` during search (e.g. ``node.Q_max`` for ``MaxValue``). The
    strategy excludes dead branches, so this agrees with ``visit_pi`` about
    which branches exist. ``None`` when no usable statistic exists yet;
    ``value_targets`` then falls back to a finite per-step default so the value
    loss never sees ``None`` / ``-inf``."""


@dataclass
class Trajectory:
    """Sequence of steps from one self-play episode (root to terminal)."""

    steps: List[TrajectoryStep]
    value_scale: Optional[float] = None
    """Positive reward cap used to map value targets into ``[-1, +1]``.
    Stamped by ``run_self_play`` from the simulator's ``reward_scale``;
    ``None`` if unknown (callers then apply their own scale / 1.0)."""

    def value_targets(self, *, value_scale: Optional[float] = None) -> List[float]:
        """Per-step value target ``z_t = clip(state_value / scale, -1, +1)``.

        ``scale`` (the positive reward cap) is taken from the argument, else
        the trajectory's stamped ``value_scale``, else ``1.0``. A step whose
        ``state_value`` is ``None`` / non-finite falls back to its own clipped
        ``reward`` (the honest chosen-step eval), or ``-1`` when that too is
        non-finite -- so the value loss never sees ``None`` / ``-inf`` / NaN.
        """
        scale = value_scale if value_scale is not None else self.value_scale
        if not scale or scale <= 0:
            scale = 1.0
        targets: List[float] = []
        for s in self.steps:
            rv = s.state_value
            if rv is not None and math.isfinite(rv):
                raw = rv
            elif math.isfinite(s.reward):
                raw = s.reward
            else:
                raw = -scale
            targets.append(max(-1.0, min(1.0, raw / scale)))
        return targets


class ReplayBuffer:
    """
    Bounded FIFO buffer of ``(state, visit_pi, value_target[, applicable_actions])`` rows.

    Args:
        capacity: max rows; oldest are evicted.
        value_scale: positive reward cap used to scale value targets into
            ``[-1, +1]`` when a pushed trajectory does not carry its own
            ``value_scale``. ``None`` -> ``1.0`` (no scaling).

    Inspect occupancy with ``len(buffer)``, ``buffer.capacity`` and
    ``buffer.fill_fraction`` -- handy for picking a capacity that actually
    evicts stale targets each iteration instead of hoarding many iterations of
    them (with <= depth_limit rows per episode, the default 10k may be far
    larger than your throughput; size it to hold a few iterations of self-play).
    """

    def __init__(self, capacity: int = 10_000, value_scale: Optional[float] = None):
        if value_scale is not None and value_scale <= 0:
            raise ValueError(f"value_scale must be > 0, got {value_scale!r}")
        self._buf: deque = deque(maxlen=capacity)
        self.value_scale = value_scale

    @property
    def capacity(self) -> Optional[int]:
        """Max rows the buffer holds (the deque's ``maxlen``)."""
        return self._buf.maxlen

    @property
    def fill_fraction(self) -> float:
        """Fraction of capacity currently used, in ``[0, 1]`` (0 if unbounded)."""
        cap = self._buf.maxlen
        return (len(self._buf) / cap) if cap else 0.0

    def push_trajectory(self, traj: Trajectory) -> None:
        """Push every step. Value targets are clipped + scaled into
        ``[-1, +1]`` via ``Trajectory.value_targets`` (the trajectory's stamped
        ``value_scale`` wins, else the buffer's). No rows are dropped.

        Each row is ``(state, visit_pi, value_target, applicable_actions)`` when
        ``step.applicable_actions`` is non-empty, else a bare 3-tuple."""
        scale = traj.value_scale if traj.value_scale is not None else self.value_scale
        zs = traj.value_targets(value_scale=scale)
        for step, z in zip(traj.steps, zs):
            if step.applicable_actions:
                self._buf.append(
                    (step.state, step.visit_pi, z, tuple(step.applicable_actions))
                )
            else:
                self._buf.append((step.state, step.visit_pi, z))

    def sample(self, batch_size: int) -> List[Tuple]:
        """Uniform random sample (without replacement) of up to ``batch_size`` rows."""
        n = min(batch_size, len(self._buf))
        if n == 0:
            return []
        return random.sample(self._buf, k=n)

    def __len__(self) -> int:
        return len(self._buf)
