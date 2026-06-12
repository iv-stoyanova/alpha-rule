"""Pytest configuration for the alpha-rule test suite.

Adds the repo root to ``sys.path`` so ``import alpha_rule`` resolves whether
the suite is run via pytest or the bundled runner. Shared fixtures are added
here as the components that need them are migrated: the fake gym env arrives
with the reinforcement-learning backend, for the observation-wrapper tests.
"""
from __future__ import annotations

import sys
from pathlib import Path

# tests/ -> alpha_rule/ -> repo root
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np  # noqa: E402
import pytest  # noqa: E402


# --------------------------------------------------------------------------- #
# Minimal fake gym env for observation-wrapper tests
# --------------------------------------------------------------------------- #
#
# ``gymnasium`` is a heavy, optional ([rl]) dependency. We delay importing it
# until a test actually constructs ``FakeGymEnv`` so the rest of the suite
# (matrix math, MCTS node, grammar, replay buffer, ...) runs even without
# ``gymnasium`` installed.

try:
    import gymnasium as _gym  # noqa: E402
    _HAS_GYMNASIUM = True
    _GymEnvBase = _gym.Env
except ImportError:                                    # pragma: no cover
    _HAS_GYMNASIUM = False
    _gym = None
    _GymEnvBase = object


class FakeGymEnv(_GymEnvBase):
    """
    Smallest surface of a gym env the observation wrappers rely on.

    Provides:
      - ``action_space`` (``Discrete(16)``) so ``len(bin(n)) - 3 == 2`` works
        in the wrapper init
      - ``get_types()`` reachable via ``env.unwrapped`` and via the ``.env``
        chain the wrapper's fallback walk uses
      - scripted ``reset`` / ``step`` emitting event dicts

    Subclasses ``gymnasium.Env`` so the ``Wrapper`` init ``isinstance`` check
    passes.
    """

    def __init__(self, events=None, event_types=("A", "B", "C")):
        if not _HAS_GYMNASIUM:
            raise RuntimeError(
                "FakeGymEnv requires gymnasium. Install with `pip install gymnasium`."
            )
        super().__init__()
        self._event_types = list(event_types)
        self._events = list(events or [])
        self._idx = 0
        self.action_space = _gym.spaces.Discrete(16)
        self.observation_space = _gym.spaces.Dict({
            "e_type": _gym.spaces.Discrete(len(self._event_types)),
            "start": _gym.spaces.Box(low=0.0, high=1e6, shape=(1,)),
            "end": _gym.spaces.Box(low=0.0, high=1e6, shape=(1,)),
        })

    # Short-circuit the chain: ``.env`` on a raw env returns self so
    # ``env.env.env.env.get_types()`` resolves here.
    @property
    def env(self):  # noqa: D401
        return self

    def get_types(self):
        return self._event_types

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._idx = 0
        obs = self._emit()
        return obs, {}

    def step(self, action):
        obs = self._emit()
        done = self._idx >= len(self._events)
        return obs, 0.0, done, False, {}

    def _emit(self):
        if not self._events:
            return {"e_type": 0, "start": np.array([0.0]), "end": np.array([1.0])}
        ev = self._events[self._idx % len(self._events)]
        self._idx += 1
        return ev


@pytest.fixture
def fake_gym_env():
    """Factory for ``FakeGymEnv`` instances."""
    def _factory(events=None, event_types=("A", "B", "C")):
        return FakeGymEnv(events=events, event_types=event_types)
    return _factory
