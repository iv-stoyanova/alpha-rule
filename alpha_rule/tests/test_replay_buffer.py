"""
Tests for ``mcts.replay``.

Value targets are the per-step ``state_value`` (from the configured
``ValueTarget``) mapped into ``[-1, +1]`` with a single linear clip:

    z = clip(state_value / value_scale, -1, +1)

``value_scale`` is the simulator's positive reward cap (small). Good rules
spread over ``[0, 1]``; the long negative tail and ``-inf`` failures collapse
to ``-1``.

Pins:
    - Each step's target = clip(state_value / scale, -1, +1).
    - A ``None`` / non-finite ``state_value`` falls back to its own clipped
      ``reward``, or ``-1`` when that too is non-finite -- never None/-inf/NaN.
    - The trajectory's stamped ``value_scale`` wins over the buffer's; an
      explicit arg to ``value_targets`` wins over both.
    - ``push_trajectory`` keeps every step; capacity evicts oldest.
    - ``capacity`` / ``fill_fraction`` report occupancy; ``value_scale`` is
      validated; ``sample`` returns ``min(len, request)``.
"""
from __future__ import annotations

import math

from alpha_rule.mcts.replay import ReplayBuffer, Trajectory, TrajectoryStep


def _step(state: str, reward: float, *, state_value=None, action: str = "A"):
    return TrajectoryStep(
        state=state, visit_pi={action: 1.0}, reward=reward, state_value=state_value,
    )


# --------------------------------------------------------------------------- #
# value_targets: clip + scale
# --------------------------------------------------------------------------- #

def test_value_targets_clip_and_scale():
    traj = Trajectory(steps=[
        _step("s0", 0.0, state_value=1.5),     # 1.5/3 = 0.5
        _step("s1", 0.0, state_value=3.0),     # 3/3   = 1.0
        _step("s2", 0.0, state_value=-9.0),    # -9/3  = -3 -> clip -1
    ], value_scale=3.0)
    assert traj.value_targets() == [0.5, 1.0, -1.0]


def test_value_targets_minus_inf_state_value_maps_to_minus_one():
    traj = Trajectory(steps=[_step("s0", -math.inf, state_value=float("-inf"))],
                      value_scale=2.0)
    assert traj.value_targets() == [-1.0]


def test_value_targets_none_falls_back_to_reward():
    traj = Trajectory(steps=[
        _step("s0", 1.0, state_value=None),           # reward 1.0/2 = 0.5
        _step("s1", -math.inf, state_value=None),     # reward -inf -> -1
    ], value_scale=2.0)
    assert traj.value_targets() == [0.5, -1.0]


def test_value_targets_default_scale_is_one():
    traj = Trajectory(steps=[_step("s0", 0.0, state_value=0.5)])   # no scale -> 1.0
    assert traj.value_targets() == [0.5]


def test_value_targets_explicit_scale_arg_overrides_stamped():
    traj = Trajectory(steps=[_step("s0", 0.0, state_value=2.0)], value_scale=4.0)
    assert traj.value_targets(value_scale=2.0) == [1.0]   # arg 2.0 wins over stamped 4.0


def test_value_targets_empty():
    assert Trajectory(steps=[]).value_targets() == []


# --------------------------------------------------------------------------- #
# ReplayBuffer
# --------------------------------------------------------------------------- #

def test_buffer_pushes_every_step_scaled():
    buf = ReplayBuffer(capacity=10, value_scale=2.0)
    buf.push_trajectory(Trajectory(steps=[
        _step("s0", 0.0, state_value=1.0),     # 0.5
        _step("s1", 0.0, state_value=-8.0),    # clip -1
    ]))
    rows = list(buf._buf)
    assert len(buf) == 2
    assert rows[0][2] == 0.5
    assert rows[1][2] == -1.0


def test_buffer_trajectory_scale_wins_over_buffer_scale():
    buf = ReplayBuffer(capacity=10, value_scale=100.0)
    traj = Trajectory(steps=[_step("s0", 0.0, state_value=1.5)], value_scale=3.0)
    buf.push_trajectory(traj)
    assert list(buf._buf)[0][2] == 0.5         # uses the trajectory's 3.0


def test_buffer_targets_in_unit_range():
    buf = ReplayBuffer(capacity=10, value_scale=2.0)
    buf.push_trajectory(Trajectory(steps=[
        _step("s0", 0.0, state_value=-500.0),
        _step("s1", 0.0, state_value=50.0),
    ]))
    for row in buf._buf:
        assert -1.0 <= row[2] <= 1.0


def test_buffer_fill_fraction_and_capacity():
    buf = ReplayBuffer(capacity=4)
    assert buf.capacity == 4
    assert buf.fill_fraction == 0.0
    buf.push_trajectory(Trajectory(steps=[
        _step("s0", 0.0, state_value=1.0),
        _step("s1", 0.0, state_value=1.0),
    ]))
    assert len(buf) == 2
    assert abs(buf.fill_fraction - 0.5) < 1e-9


def test_buffer_capacity_evicts_oldest():
    buf = ReplayBuffer(capacity=3, value_scale=10.0)
    for r in [1.0, 2.0, 3.0, 4.0, 5.0]:
        buf.push_trajectory(Trajectory(steps=[_step(f"s{r}", 0.0, state_value=r)]))
    assert len(buf) == 3
    assert [row[0] for row in buf._buf] == ["s3.0", "s4.0", "s5.0"]
    assert abs(buf.fill_fraction - 1.0) < 1e-9


def test_buffer_sample_returns_min_of_len_and_request():
    buf = ReplayBuffer(capacity=10, value_scale=10.0)
    for r in [1.0, 2.0]:
        buf.push_trajectory(Trajectory(steps=[_step(f"s{r}", 0.0, state_value=r)]))
    assert len(buf.sample(batch_size=10)) == 2
    assert len(buf.sample(batch_size=1)) == 1


def test_buffer_stores_applicable_actions_as_fourth_element():
    buf = ReplayBuffer(capacity=10, value_scale=10.0)
    step = TrajectoryStep(
        state="<ROOT>", visit_pi={"A": 1.0}, reward=1.0, state_value=5.0,
        applicable_actions=("A", "B"),
    )
    buf.push_trajectory(Trajectory(steps=[step]))
    row = list(buf._buf)[0]
    assert len(row) == 4 and row[3] == ("A", "B")


def test_targets_are_finite_floats():
    buf = ReplayBuffer(capacity=4, value_scale=2.0)
    buf.push_trajectory(Trajectory(steps=[
        _step("s0", -math.inf, state_value=None),
        _step("s1", 1.0, state_value=2.0),
    ]))
    for row in buf._buf:
        z = row[2]
        assert isinstance(z, float) and math.isfinite(z)


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #

def test_buffer_rejects_nonpositive_scale():
    import pytest
    with pytest.raises(ValueError):
        ReplayBuffer(capacity=10, value_scale=0.0)
    with pytest.raises(ValueError):
        ReplayBuffer(capacity=10, value_scale=-1.0)
