"""
Running reward normalizer used at read time by the MCTS search.

The tree stores raw rewards (``backprop`` is untouched). This maps a reward into
``[-1, 1]`` (recentered, scaled) at the points where a reward magnitude is read:
``PUCTSelection.score``, ``Trajectory.value_targets``, and the ``NeuralEvaluator``
de-scale. ``normalize`` and ``denormalize`` are inverses for finite values, so the
value target and the network de-scale share one scale.

Center/scale come from one of two estimators:
    robust=True  (default): median + 1.4826 * MAD. The robust scale ignores the
        extreme negative tail (bad rules), so the competitive band is not crushed
        into a sliver of [-1, 1].
    robust=False: Welford running mean + population std (kept for comparison and
        for the exact-arithmetic tests).

Stats are accumulated online over finite values only; ``-inf`` (the dead-marking
sentinel) and other non-finite values pass through unchanged.
"""
from __future__ import annotations

import math
from typing import List, Optional


def _median(sorted_vals: List[float]) -> float:
    m = len(sorted_vals)
    mid = m // 2
    if m % 2:
        return sorted_vals[mid]
    return 0.5 * (sorted_vals[mid - 1] + sorted_vals[mid])


class RewardNormalizer:
    """Online center/scale for read-time reward normalization.

        normalize(q)   = clip((q - center) / (k * scale), -1, 1)
        denormalize(v) = v * (k * scale) + center

    ``k`` (passed by the caller, default ``2.0`` in the training config) sets how
    many scale units map to the clip edge. ``scale`` is floored at ``eps`` and
    reported as ``1.0`` until two finite samples exist.

    Args:
        eps: floor for the scale divisor.
        robust: when ``True`` (default) center/scale are the median and
            ``1.4826 * MAD`` over the finite rewards seen this round; when
            ``False`` they are the Welford mean and population std.
        recompute_every: under ``robust``, recompute the cached median/MAD only
            once per this many updates (reads are O(1) off the cache).
    """

    def __init__(self, eps: float = 1e-6, robust: bool = True,
                 recompute_every: int = 16):
        self.eps = eps
        self.robust = robust
        self.recompute_every = max(1, recompute_every)
        self.count = 0
        # Welford accumulators (used when robust is False).
        self._mean = 0.0
        self._M2 = 0.0
        # Robust state: the finite values this round + a cached center/scale.
        self._values: List[float] = []
        self._center = 0.0
        self._scale = 1.0
        self._computed_at = -1

    def update(self, value: float) -> None:
        """Fold one reward into the running stats. Non-finite values are skipped."""
        if value is None or not math.isfinite(value):
            return
        self.count += 1
        delta = value - self._mean
        self._mean += delta / self.count
        self._M2 += delta * (value - self._mean)
        if self.robust:
            self._values.append(value)

    def reset(self) -> None:
        """Clear the stats (called once per discovery round)."""
        self.count = 0
        self._mean = 0.0
        self._M2 = 0.0
        self._values = []
        self._center = 0.0
        self._scale = 1.0
        self._computed_at = -1

    def _maybe_recompute(self) -> None:
        """Refresh the cached robust center/scale at most once per
        ``recompute_every`` updates."""
        n = len(self._values)
        if n < 2:
            self._center = self._values[0] if n == 1 else 0.0
            self._scale = 1.0
            return
        # Refresh promptly while the sample is small, throttle to recompute_every
        # once it is large (keeps reads O(1) between refreshes).
        threshold = min(self.recompute_every, max(1, n // 8))
        if self._computed_at >= 0 and (n - self._computed_at) < threshold:
            return
        vals = sorted(self._values)
        med = _median(vals)
        mad = _median(sorted(abs(v - med) for v in vals))
        self._center = med
        self._scale = max(1.4826 * mad, self.eps)
        self._computed_at = n

    @property
    def mean(self) -> float:
        """Distribution center: median (robust) or running mean (Welford)."""
        if not self.robust:
            return self._mean
        self._maybe_recompute()
        return self._center

    @property
    def std(self) -> float:
        """Distribution scale floored at ``eps``; ``1.0`` until two finite
        samples exist. Robust scale is ``1.4826 * MAD``; otherwise population
        std."""
        if not self.robust:
            if self.count < 2:
                return 1.0
            return max(math.sqrt(self._M2 / self.count), self.eps)
        self._maybe_recompute()
        return self._scale

    def normalize(self, q: Optional[float], k: float = 2.0):
        """Map a raw reward into ``[-1, 1]``. ``None`` and non-finite values pass
        through unchanged (keeps the ``-inf`` sentinel and ``None``)."""
        if q is None or not math.isfinite(q):
            return q
        scale = k * self.std
        if scale <= 0.0:
            scale = self.eps
        z = (q - self.mean) / scale
        if z > 1.0:
            return 1.0
        if z < -1.0:
            return -1.0
        return z

    def denormalize(self, v: float, k: float = 2.0) -> float:
        """Inverse of ``normalize`` for finite ``v`` (no clip)."""
        return v * (k * self.std) + self.mean
