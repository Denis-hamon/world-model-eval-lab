"""Unit tests for the CEMPlanner adapter."""

from __future__ import annotations

import pytest

from wmel.adapters.cem_planner import CEMPlanner


def _open_grid_dynamics(state: tuple[int, int], action: str) -> tuple[int, int]:
    x, y = state
    if action == "up":
        return (x, y + 1)
    if action == "down":
        return (x, y - 1)
    if action == "right":
        return (x + 1, y)
    if action == "left":
        return (x - 1, y)
    raise ValueError(action)


ACTIONS = ("up", "down", "left", "right")


def test_encode_is_identity() -> None:
    planner = CEMPlanner(dynamics=_open_grid_dynamics, action_space=ACTIONS, seed=0)
    assert planner.encode((3, 4)) == (3, 4)


def test_rollout_walks_through_dynamics() -> None:
    planner = CEMPlanner(dynamics=_open_grid_dynamics, action_space=ACTIONS, seed=0)
    trajectory = planner.rollout((0, 0), ["up", "up", "right"])
    assert trajectory == [(0, 0), (0, 1), (0, 2), (1, 2)]


def test_compute_per_plan_call_is_iters_x_samples_x_horizon() -> None:
    planner = CEMPlanner(
        dynamics=_open_grid_dynamics,
        action_space=ACTIONS,
        num_iterations=4,
        num_samples=20,
        num_elites=5,
        plan_horizon=12,
        seed=0,
    )
    assert planner.compute_per_plan_call == 4 * 20 * 12


def test_plan_reaches_goal_in_open_grid() -> None:
    planner = CEMPlanner(
        dynamics=_open_grid_dynamics,
        action_space=ACTIONS,
        num_iterations=4,
        num_samples=48,
        num_elites=8,
        plan_horizon=12,
        smoothing=0.1,
        seed=0,
    )
    actions = planner.plan(observation=(0, 0), goal=(3, 3), horizon=12)
    pos: tuple[int, int] = (0, 0)
    for a in actions:
        pos = _open_grid_dynamics(pos, a)
    assert pos == (3, 3)


def test_plan_empty_when_horizon_zero() -> None:
    planner = CEMPlanner(dynamics=_open_grid_dynamics, action_space=ACTIONS, seed=0)
    assert planner.plan(observation=(0, 0), goal=(1, 1), horizon=0) == []


def test_invalid_args_raise() -> None:
    with pytest.raises(ValueError):
        CEMPlanner(dynamics=_open_grid_dynamics, action_space=ACTIONS, num_iterations=0)
    with pytest.raises(ValueError):
        CEMPlanner(dynamics=_open_grid_dynamics, action_space=ACTIONS, num_samples=0)
    with pytest.raises(ValueError):
        CEMPlanner(dynamics=_open_grid_dynamics, action_space=ACTIONS, num_elites=0)
    with pytest.raises(ValueError):
        CEMPlanner(dynamics=_open_grid_dynamics, action_space=ACTIONS, num_samples=5, num_elites=6)
    with pytest.raises(ValueError):
        CEMPlanner(dynamics=_open_grid_dynamics, action_space=ACTIONS, plan_horizon=0)
    with pytest.raises(ValueError):
        CEMPlanner(dynamics=_open_grid_dynamics, action_space=ACTIONS, smoothing=1.5)
    with pytest.raises(ValueError):
        CEMPlanner(dynamics=_open_grid_dynamics, action_space=(), num_iterations=1, num_samples=2, num_elites=1)
