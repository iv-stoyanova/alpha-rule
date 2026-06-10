"""
Replay buffer + value-target computation for AlphaZero-style training.

The data flow:

    self_play.run_self_play() produces Trajectory(steps)
                                       |
                       value_targets = clip(MCTS root value at s_t)
                                       |
    ReplayBuffer.push_trajectory(traj) writes (state, π_t, z_t) rows
                                       |
                buffer.sample(batch_size) -> NN training step

``-inf`` policy
----------------
``RuleSimulator.evaluate`` returns ``-inf`` when a candidate rule never
matches any episode in the environment. That's a structural failure
("the formula is unrealisable"), not a low score. We want the value
head to LEARN to avoid such productions, but feeding ``-inf`` into MSE
would blow up the loss.

So we clip ``-inf`` rewards to a configurable ``reward_floor`` (default
``-100.0``, well below any achievable finite reward) BEFORE computing
the suffix-max. The buffer keeps every step, failures become strong
negative training signal rather than missing data.
"""
from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple


DEFAULT_REWARD_FLOOR: float = -100.0
"""Default value used to replace ``-inf`` rewards before training.
Doubles as the negative end of the symmetric reward-clip range."""

DEFAULT_REWARD_CEILING: float = 100.0
"""Default upper bound for reward clipping. Together with
``DEFAULT_REWARD_FLOOR`` defines the ``[-100, +100]`` clip range; with
the default ``value_target_scale = max(|floor|, |ceiling|) = 100`` this
maps stored value targets into ``[-1, +1]``, matching the tanh-bounded
``ValueHead`` output range."""


@dataclass
class TrajectoryStep:
    """
    One MCTS root-step inside a self-play episode.

    Follows the AlphaZero paper's tuple ``(s_t, π_t, R(s_{t+1}))``:
    ``state`` is the state **from which** MCTS was rooted and π_t
    computed, but ``reward`` is the reward of the **child** we then
    advanced to. The two fields are semantically offset by one step.

    ``next_state`` records the child's name so "best formula" / UI
    code has a handle on the rule the reward actually describes
    (otherwise ``_best_in_trajectory`` surfaces the parent
    ``"<ROOT>"`` whenever the first action happened to be the best).
    """

    state: object
    """The state the policy π_t was computed at. Today an
    ``MCTSRuleNode`` or its name string."""

    visit_pi: Dict[str, float]
    """Production-name -> normalised MCTS visit count at ``state``."""

    reward: float
    """R(next_state), reward of the child we advanced to after sampling
    from ``visit_pi``. May be ``-inf`` for failed evaluations; the
    replay buffer clips it to ``reward_floor`` before training."""

    next_state: Optional[object] = None
    """Name/identity of the child that earned ``reward``. Populated by
    ``run_self_play``; ``None`` for older code paths that only recorded
    the state/reward offset pair."""

    applicable_actions: Tuple[str, ...] = ()
    """Production names that were legal at ``state`` per the grammar.
    Used by the training-time softmax mask so the loss denominator
    matches ``NeuralEvaluator``'s inference-time mask. Empty tuple
    (default) means "no mask", preserves unmasked training for
    code paths that don't supply it."""

    root_value: Optional[float] = None
    """MCTS root value at ``state`` after the search round that
    produced ``visit_pi``. Defined as the visit-weighted average of
    children's ``Q_max`` (with dead children's ``-inf`` replaced by
    ``reward_floor`` so the average stays finite and the value head
    learns to penalise states with dead branches).

    ``None`` when no usable statistic exists yet (e.g., subtree died
    before any rollout produced a finite Q). Consumed by
    ``Trajectory.value_targets(mode="root_value")``; the fallback path
    there guarantees no ``-inf``/``NaN`` reaches the NN's MSE loss."""


@dataclass
class Trajectory:
    """Sequence of steps from one self-play episode (root to terminal)."""

    steps: List[TrajectoryStep]

    def value_targets(
        self,
        *,
        reward_floor: float = DEFAULT_REWARD_FLOOR,
        reward_ceiling: float = DEFAULT_REWARD_CEILING,
    ) -> List[float]:
        """
        Per-step value target ``z_t`` = the MCTS root value computed during
        search at ``step.state`` (the visit-weighted average of children's
        ``Q_max``), clipped to ``[reward_floor, reward_ceiling]``. ``-inf``
        lands at ``reward_floor``; overshooting positives pin at
        ``reward_ceiling``. The two-sided clip keeps targets bounded so they
        normalise into ``[-1, +1]`` for the tanh value head.

        This is the ExIt / AlphaGo-Zero recipe for single-agent problems
        with informative intermediate states: per-step targets differ
        between states, so the value head learns to tell them apart.

        Per-step fallback when ``step.root_value`` is ``None`` / non-finite
        (e.g. the subtree died before any rollout produced a finite Q): the
        trajectory's first finite clipped chosen-step reward, guaranteeing
        no ``-inf`` / ``NaN`` ever reaches the MSE loss.
        """
        if not self.steps:
            return []

        # Per-step fallback: trajectory's first finite chosen-step reward
        # (clipped). Stable + finite once any step produced a finite reward.
        fallback = float(reward_floor)
        for s in self.steps:
            if math.isfinite(s.reward):
                fallback = max(float(reward_floor), min(float(reward_ceiling), s.reward))
                break

        targets: List[float] = []
        for s in self.steps:
            rv = s.root_value
            if rv is not None and math.isfinite(rv):
                t = max(float(reward_floor), min(float(reward_ceiling), rv))
            else:
                t = fallback
            targets.append(t)
        return targets


class ReplayBuffer:
    """
    Bounded FIFO buffer of ``(state, visit_pi, value_target)`` rows.

    Args:
        capacity: max rows. Older rows are evicted.
        reward_floor: lower clip + replacement for ``-inf`` rewards
            during target computation.
        reward_ceiling: upper clip for overshooting positive rewards.
            Together with ``reward_floor`` defines the bounded target
            range.
        value_target_scale: divisor applied to every stored value target.
            Default (= ``max(|reward_floor|, |reward_ceiling|)``) maps
            stored targets into ``[-1, +1]``, matching the tanh-bounded
            ``ValueHead`` output range. Pass ``1.0`` explicitly to opt
            into raw-reward-scale targets (the value head will then
            saturate). ``NeuralEvaluator`` should be built with a matching
            ``value_scale`` so its predictions are multiplied back up to
            raw-reward units for MCTS backup.

    Value targets are the per-step MCTS root value (ExIt / AlphaGo-Zero);
    see ``Trajectory.value_targets``.
    """

    def __init__(
        self,
        capacity: int = 10_000,
        reward_floor: float = DEFAULT_REWARD_FLOOR,
        reward_ceiling: float = DEFAULT_REWARD_CEILING,
        value_target_scale: Optional[float] = None,
    ):
        if reward_ceiling <= reward_floor:
            raise ValueError(
                f"reward_ceiling ({reward_ceiling!r}) must exceed "
                f"reward_floor ({reward_floor!r})"
            )
        if value_target_scale is None:
            value_target_scale = max(abs(reward_floor), abs(reward_ceiling))
        if not (value_target_scale > 0):
            raise ValueError(
                f"value_target_scale must be > 0, got {value_target_scale!r}"
            )
        self._buf: deque = deque(maxlen=capacity)
        self.reward_floor = reward_floor
        self.reward_ceiling = reward_ceiling
        self.value_target_scale = float(value_target_scale)

    def push_trajectory(self, traj: Trajectory) -> None:
        """
        Push every step in ``traj``. ``-inf`` rewards become
        ``reward_floor`` in the value target via ``Trajectory.value_targets``;
        no rows are dropped. Targets are divided by ``value_target_scale``
        before storage.

        Each row is a 4-tuple ``(state, visit_pi, value_target,
        applicable_actions)`` when ``step.applicable_actions`` is
        non-empty, otherwise a bare 3-tuple ``(state, visit_pi,
        value_target)``. Train-side ``collate`` accepts both shapes.
        """
        zs = traj.value_targets(
            reward_floor=self.reward_floor,
            reward_ceiling=self.reward_ceiling,
        )
        scale = self.value_target_scale
        for step, z in zip(traj.steps, zs):
            scaled_z = float(z) / scale
            if step.applicable_actions:
                self._buf.append(
                    (step.state, step.visit_pi, scaled_z, tuple(step.applicable_actions))
                )
            else:
                self._buf.append((step.state, step.visit_pi, scaled_z))

    def sample(self, batch_size: int) -> List[Tuple]:
        """Uniform random sample (without replacement) of up to ``batch_size`` rows."""
        n = min(batch_size, len(self._buf))
        if n == 0:
            return []
        return random.sample(self._buf, k=n)

    def __len__(self) -> int:
        return len(self._buf)
