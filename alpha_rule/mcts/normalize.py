"""
Running reward normalizer used at read time by the MCTS search.

The tree stores raw rewards (``backprop`` is untouched). This maps a reward into
``[-1, 1]`` (recentered, scaled by the running std) at the points where a reward
magnitude is read: ``PUCTSelection.score``, ``Trajectory.value_targets``, and the
``NeuralEvaluator`` de-scale. ``normalize`` and ``denormalize`` are inverses for
finite values, so the value target and the network de-scale share one scale.

Stats are accumulated online (Welford) over finite values only; ``-inf`` (the
dead-marking sentinel) and other non-finite values pass through unchanged.
"""
from __future__ import annotations

import math
from typing import Optional


class RewardNormalizer:
    """Online running mean/std for read-time reward normalization.

        normalize(q)   = clip((q - mean) / (k * std), -1, 1)
        denormalize(v) = v * (k * std) + mean

    ``k`` (passed by the caller, default ``2.0`` in the training config) sets how
    many std map to the clip edge. ``std`` is floored at ``eps`` and reported as
    ``1.0`` until two finite samples exist.

    Args:
        eps: floor for the standard-deviation divisor.
    """

    def __init__(self, eps: float = 1e-6):
        self.eps = eps
        self.count = 0
        self._mean = 0.0
        self._M2 = 0.0

    def update(self, value: float) -> None:
        """Fold one reward into the running stats. Non-finite values are skipped."""
        if value is None or not math.isfinite(value):
            return
        self.count += 1
        delta = value - self._mean
        self._mean += delta / self.count
        self._M2 += delta * (value - self._mean)

    def reset(self) -> None:
        """Clear the stats (called once per discovery round)."""
        self.count = 0
        self._mean = 0.0
        self._M2 = 0.0

    @property
    def mean(self) -> float:
        return self._mean

    @property
    def std(self) -> float:
        """Population std floored at ``eps``; ``1.0`` until two finite samples
        exist, so a single sample does not drive the scale to zero."""
        if self.count < 2:
            return 1.0
        return max(math.sqrt(self._M2 / self.count), self.eps)

    def normalize(self, q: Optional[float], k: float = 2.0):
        """Map a raw reward into ``[-1, 1]``. ``None`` and non-finite values pass
        through unchanged (keeps the ``-inf`` sentinel and ``None``)."""
        if q is None or not math.isfinite(q):
            return q
        scale = k * self.std
        if scale <= 0.0:
            scale = self.eps
        z = (q - self._mean) / scale
        if z > 1.0:
            return 1.0
        if z < -1.0:
            return -1.0
        return z

    def denormalize(self, v: float, k: float = 2.0) -> float:
        """Inverse of ``normalize`` for finite ``v`` (no clip)."""
        return v * (k * self.std) + self._mean
