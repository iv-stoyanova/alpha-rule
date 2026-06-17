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
from dataclasses import dataclass, field
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
    neg_value_scale: Optional[float] = None
    """Negative reward cap for ASYMMETRIC value-target scaling: rewards below 0
    are divided by this instead of ``value_scale``. ``None`` -> falls back to
    ``value_scale`` (symmetric, the historical behaviour). Set it larger than
    ``value_scale`` when the positive cap is small (e.g. +3 for 3 boxes) but the
    penalty tail runs far more negative, so good rules keep full resolution in
    ``[0, 1]`` while bad rules spread across ``[-1, 0)`` instead of all
    saturating at ``-1``."""
    norm_mean: Optional[float] = None
    """Running reward mean stamped by ``run_self_play`` when read-time
    normalization is ON. When ``norm_mean`` and ``norm_std`` are both set,
    ``value_targets`` uses the recentered single-scale mapping
    ``z = clip((raw - norm_mean) / (norm_k * norm_std), -1, +1)`` instead of the
    asymmetric ``value_scale``/``neg_value_scale`` split. ``None`` -> the
    asymmetric mapping (historical behaviour)."""
    norm_std: Optional[float] = None
    """Running reward std for the recentered mapping (see ``norm_mean``)."""
    norm_k: Optional[float] = None
    """Std-per-unit for the recentered mapping: a reward ``norm_k`` std above the
    mean maps to ``+1`` before the clip. ``None`` -> ``2.0`` when normalizing."""
    dead_names: List[str] = field(default_factory=list)
    """Rule names killed during this episode because their ``<END>`` scored
    ``-inf`` (the rule never fires, so its whole subtree is dead). ``train``
    folds these into its persistent dead set so future iterations prune them
    without re-spending a simulator call. Empty for episodes that killed
    nothing."""

    def value_targets(
        self,
        *,
        value_scale: Optional[float] = None,
        neg_value_scale: Optional[float] = None,
        norm_mean: Optional[float] = None,
        norm_std: Optional[float] = None,
        norm_k: Optional[float] = None,
    ) -> List[float]:
        """Per-step value target in ``[-1, +1]``.

        Read-time normalization (``norm_mean`` and ``norm_std`` set, from the
        args or the stamped fields) uses one recentered scale:

            z = clip((raw - norm_mean) / (norm_k * norm_std), -1, +1)

        the exact inverse of ``NeuralEvaluator.denormalize``. Otherwise the
        asymmetric split applies:

            z = min(+1, raw / pos)   for raw >= 0
            z = max(-1, raw / neg)   for raw < 0

        ``pos`` is the positive cap (arg, else stamped ``value_scale``, else
        ``1.0``); ``neg`` is the negative cap (arg, else stamped
        ``neg_value_scale``, else ``pos`` -> symmetric). A step whose
        ``state_value`` is ``None`` / non-finite falls back to its own
        ``reward``, or to the clip floor ``-1`` when that too is non-finite, so
        the value loss never sees ``None`` / ``-inf`` / NaN.
        """
        nm = norm_mean if norm_mean is not None else self.norm_mean
        ns = norm_std if norm_std is not None else self.norm_std
        nk = norm_k if norm_k is not None else self.norm_k
        if nk is None:
            nk = 2.0
        use_norm = nm is not None and ns is not None
        norm_scale = (nk * ns) if use_norm else None
        if use_norm and (norm_scale is None or norm_scale <= 0):
            norm_scale = 1e-6

        pos = value_scale if value_scale is not None else self.value_scale
        if not pos or pos <= 0:
            pos = 1.0
        neg = neg_value_scale if neg_value_scale is not None else self.neg_value_scale
        if not neg or neg <= 0:
            neg = pos

        targets: List[float] = []
        for s in self.steps:
            rv = s.state_value
            if rv is not None and math.isfinite(rv):
                raw = rv
            elif math.isfinite(s.reward):
                raw = s.reward
            else:
                raw = None                          # non-finite -> clip floor
            if use_norm:
                if raw is None:
                    targets.append(-1.0)
                else:
                    z = (raw - nm) / norm_scale
                    targets.append(max(-1.0, min(1.0, z)))
            else:
                if raw is None:
                    raw = -neg
                if raw >= 0:
                    targets.append(min(1.0, raw / pos))
                else:
                    targets.append(max(-1.0, raw / neg))
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

    def __init__(
        self,
        capacity: int = 10_000,
        value_scale: Optional[float] = None,
        neg_value_scale: Optional[float] = None,
    ):
        if value_scale is not None and value_scale <= 0:
            raise ValueError(f"value_scale must be > 0, got {value_scale!r}")
        if neg_value_scale is not None and neg_value_scale <= 0:
            raise ValueError(
                f"neg_value_scale must be > 0, got {neg_value_scale!r}"
            )
        self._buf: deque = deque(maxlen=capacity)
        self.value_scale = value_scale
        self.neg_value_scale = neg_value_scale

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
        neg = traj.neg_value_scale if traj.neg_value_scale is not None else self.neg_value_scale
        zs = traj.value_targets(value_scale=scale, neg_value_scale=neg)
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
