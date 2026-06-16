"""One-hot box-action wrapper for OpenTheChests-style discrete envs.

The base env decodes its integer action as an ``n_boxes``-bit button mask
(MSB = box 0), so the raw action space is ``Discrete(2**n_boxes)`` and a single
integer can press several boxes at once. For the rule-search agent we only ever
want "press exactly one box, or none", which both shrinks the action space (from
``2**n`` to ``n + 1``) and removes compound wrong-presses.

``OneHotBoxActionWrapper`` exposes ``Discrete(n_boxes + 1)`` and maps the agent's
action to the env's integer mask:

    0        -> 0              (press nothing)
    i (1..n) -> 1 << (n - i)   (open only box ``i - 1``, MSB-first)

So for 3 boxes the agent's ``{0, 1, 2, 3}`` become env actions ``{0, 4, 2, 1}``
(``000, 100, 010, 001``). The mapping is derived from the box count, not
hardcoded, so it scales to any ``n``.

Because the mapping lives in one wrapper, training and evaluation both step the
env with the agent's raw action and get the same interpretation -- no per-call
remapping in the builder or the evaluators.
"""
from __future__ import annotations

import gymnasium as gym


class OneHotBoxActionWrapper(gym.ActionWrapper):
    """Restrict an ``Discrete(2**n_boxes)`` mask action space to a one-hot
    "open one box or none" choice of size ``n_boxes + 1``.

    Args:
        env: an env (possibly already observation-wrapped) whose
            ``action_space`` is ``Discrete(2**n_boxes)`` -- the OTC button mask.

    Raises:
        ValueError: if the wrapped action space is not ``Discrete`` of a
            power-of-two size (so a one-hot box mapping is well-defined).
    """

    def __init__(self, env):
        super().__init__(env)
        if not isinstance(env.action_space, gym.spaces.Discrete):
            raise ValueError(
                "OneHotBoxActionWrapper needs a Discrete action space, got "
                f"{type(env.action_space).__name__}."
            )
        full_n = int(env.action_space.n)
        # OTC actions are an n-bit mask: full_n == 2**n_boxes. Reject anything
        # that is not a power of two so ``n_boxes`` is unambiguous.
        if full_n < 2 or (full_n & (full_n - 1)) != 0:
            raise ValueError(
                "OneHotBoxActionWrapper expects a Discrete(2**n_boxes) action "
                f"space (a power-of-two button mask); got Discrete({full_n})."
            )
        self.n_boxes = full_n.bit_length() - 1
        self.action_space = gym.spaces.Discrete(self.n_boxes + 1)

    def action(self, action):
        """Map a one-hot agent action to the env's integer button mask.

        ``0`` presses nothing; ``i`` in ``1..n_boxes`` opens only box ``i - 1``
        (``1 << (n_boxes - i)``, MSB-first).
        """
        a = int(action)
        if a == 0:
            return 0
        return 1 << (self.n_boxes - a)
