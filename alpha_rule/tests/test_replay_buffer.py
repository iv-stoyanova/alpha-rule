"""
Tests for ``mcts.replay``.

Value targets are the per-step MCTS root value (ExIt / AlphaGo-Zero),
clipped to ``[reward_floor, reward_ceiling]`` then divided by
``value_target_scale``.

Pins:
    - Each step's target = its (clipped) ``root_value``.
    - ``root_value`` that is ``None`` / non-finite falls back per-step to
      the trajectory's first finite clipped chosen-step reward, so the NN
      never sees ``-inf`` / ``NaN``.
    - ``ReplayBuffer.push_trajectory`` keeps every step (no drops) and
      scales every stored target; ``-inf`` lands at ``reward_floor``.
    - ``reward_floor`` / ``reward_ceiling`` / ``value_target_scale`` are
      configurable and validated; capacity evicts oldest; ``sample``
      returns ``min(capacity, request)``.
"""
from __future__ import annotations

import math

from alpha_rule.mcts.replay import (
    DEFAULT_REWARD_FLOOR,
    ReplayBuffer,
    Trajectory,
    TrajectoryStep,
)


def _step(state: str, reward: float, *, root_value=None, action: str = "A"):
    return TrajectoryStep(
        state=state, visit_pi={action: 1.0}, reward=reward, root_value=root_value,
    )


# --------------------------------------------------------------------------- #
# value_targets: per-step root value + clip
# --------------------------------------------------------------------------- #

def test_value_targets_per_step_root_value():
    traj = Trajectory(steps=[
        _step("s0", 0.1, root_value=25.0),
        _step("s1", 0.5, root_value=50.0),
        _step("s2", 0.3, root_value=45.0),
    ])
    assert traj.value_targets() == [25.0, 50.0, 45.0]


def test_value_targets_clipped_to_range():
    traj = Trajectory(steps=[
        _step("s0", 0.0, root_value=-500.0),     # -> floor
        _step("s1", 0.0, root_value=500.0),      # -> ceiling
    ])
    z = traj.value_targets(reward_floor=-100.0, reward_ceiling=100.0)
    assert z == [-100.0, 100.0]


def test_value_targets_none_root_value_falls_back_to_first_finite_reward():
    traj = Trajectory(steps=[
        _step("s0", 0.7, root_value=None),        # fallback = 0.7
        _step("s1", -math.inf, root_value=40.0),  # uses its own root_value
        _step("s2", 0.3, root_value=None),        # fallback again = 0.7
    ])
    z = traj.value_targets()
    assert z == [0.7, 40.0, 0.7]


def test_value_targets_all_failed_uses_floor():
    traj = Trajectory(steps=[
        _step("s0", -math.inf, root_value=None),
        _step("s1", -math.inf, root_value=None),
    ])
    z = traj.value_targets()
    assert z == [DEFAULT_REWARD_FLOOR, DEFAULT_REWARD_FLOOR]
    assert all(math.isfinite(t) for t in z)


def test_value_targets_non_finite_root_value_falls_back():
    traj = Trajectory(steps=[
        _step("s0", 30.0, root_value=float("nan")),
        _step("s1", 0.0, root_value=float("-inf")),
        _step("s2", 0.0, root_value=float("inf")),
    ])
    z = traj.value_targets()
    # All fall back to the first finite chosen-step reward = 30.0.
    assert z == [30.0, 30.0, 30.0]
    assert all(math.isfinite(t) for t in z)


def test_value_targets_empty_trajectory():
    assert Trajectory(steps=[]).value_targets() == []


# --------------------------------------------------------------------------- #
# ReplayBuffer: keeps every step, clips + scales on write
# --------------------------------------------------------------------------- #

def test_buffer_pushes_every_step():
    buf = ReplayBuffer(capacity=10)
    traj = Trajectory(steps=[
        _step("s0", -math.inf, root_value=10.0),
        _step("s1", 5.0, root_value=20.0),
        _step("s2", -math.inf, root_value=None),
    ])
    buf.push_trajectory(traj)
    assert len(buf) == 3


def test_buffer_default_scale_auto_derives_from_reward_range():
    buf = ReplayBuffer(capacity=10)
    assert buf.value_target_scale == 100.0          # max(|-100|, |+100|)
    assert buf.reward_ceiling == 100.0
    buf.push_trajectory(Trajectory(steps=[
        _step("s0", 0.0, root_value=80.0),
        _step("s1", 0.0, root_value=20.0),
    ]))
    rows = list(buf._buf)
    assert abs(rows[0][2] - 0.80) < 1e-9            # 80 / 100
    assert abs(rows[1][2] - 0.20) < 1e-9


def test_buffer_explicit_scale_one_preserves_raw_targets():
    buf = ReplayBuffer(capacity=10, value_target_scale=1.0)
    buf.push_trajectory(Trajectory(steps=[_step("s0", 0.0, root_value=5.0)]))
    assert list(buf._buf)[0][2] == 5.0


def test_buffer_minus_inf_root_value_lands_at_floor_over_scale():
    buf = ReplayBuffer(capacity=10, reward_floor=-100.0, value_target_scale=100.0)
    # root_value None + reward -inf -> fallback floor -> -100 / 100 = -1.0
    buf.push_trajectory(Trajectory(steps=[_step("s0", -math.inf, root_value=None)]))
    assert list(buf._buf)[0][2] == -1.0


def test_buffer_targets_after_clip_and_scale_are_in_unit_range():
    buf = ReplayBuffer(capacity=10)
    buf.push_trajectory(Trajectory(steps=[
        _step("s0", 0.0, root_value=-500.0),
        _step("s1", 0.0, root_value=50.0),
        _step("s2", 0.0, root_value=1000.0),
    ]))
    for _state, _pi, z in buf._buf:
        assert -1.0 <= z <= 1.0


def test_buffer_capacity_evicts_oldest():
    buf = ReplayBuffer(capacity=3)
    for r in [1.0, 2.0, 3.0, 4.0, 5.0]:
        buf.push_trajectory(Trajectory(steps=[_step(f"s{r}", 0.0, root_value=r)]))
    assert len(buf) == 3
    assert [row[0] for row in buf._buf] == ["s3.0", "s4.0", "s5.0"]


def test_buffer_sample_returns_min_of_capacity_and_request():
    buf = ReplayBuffer(capacity=10)
    for r in [1.0, 2.0]:
        buf.push_trajectory(Trajectory(steps=[_step(f"s{r}", 0.0, root_value=r)]))
    assert len(buf.sample(batch_size=10)) == 2
    assert len(buf.sample(batch_size=1)) == 1


def test_buffer_stores_applicable_actions_as_fourth_tuple_element():
    buf = ReplayBuffer(capacity=10)
    step = TrajectoryStep(
        state="<ROOT>", visit_pi={"A": 1.0}, reward=1.0, root_value=5.0,
        applicable_actions=("A", "B"),
    )
    buf.push_trajectory(Trajectory(steps=[step]))
    row = list(buf._buf)[0]
    assert len(row) == 4 and row[3] == ("A", "B")


def test_targets_are_finite_floats():
    buf = ReplayBuffer(capacity=4)
    buf.push_trajectory(Trajectory(steps=[
        _step("s0", -math.inf, root_value=None),
        _step("s1", 1.0, root_value=2.0),
    ]))
    for row in buf._buf:
        z = row[2]
        assert isinstance(z, float) and math.isfinite(z)


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #

def test_buffer_scale_rejects_nonpositive():
    import pytest
    with pytest.raises(ValueError):
        ReplayBuffer(capacity=10, value_target_scale=0.0)
    with pytest.raises(ValueError):
        ReplayBuffer(capacity=10, value_target_scale=-1.0)


def test_buffer_rejects_ceiling_at_or_below_floor():
    import pytest
    with pytest.raises(ValueError):
        ReplayBuffer(capacity=10, reward_floor=0.0, reward_ceiling=-1.0)
    with pytest.raises(ValueError):
        ReplayBuffer(capacity=10, reward_floor=5.0, reward_ceiling=5.0)
