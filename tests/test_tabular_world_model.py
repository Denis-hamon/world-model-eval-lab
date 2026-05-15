"""Unit tests for the TabularWorldModelPlanner adapter."""

from __future__ import annotations

import pytest

from wmel.adapters.tabular_world_model import TabularWorldModelPlanner


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
    planner = TabularWorldModelPlanner(dynamics=_open_grid_dynamics, action_space=ACTIONS, seed=0)
    assert planner.encode((3, 4)) == (3, 4)


def test_rollout_walks_through_dynamics() -> None:
    planner = TabularWorldModelPlanner(dynamics=_open_grid_dynamics, action_space=ACTIONS, seed=0)
    trajectory = planner.rollout((0, 0), ["up", "up", "right"])
    assert trajectory == [(0, 0), (0, 1), (0, 2), (1, 2)]


def test_score_is_manhattan_by_default() -> None:
    planner = TabularWorldModelPlanner(dynamics=_open_grid_dynamics, action_space=ACTIONS, seed=0)
    assert planner.score((0, 0), (3, 4)) == 7.0


def test_plan_reaches_goal_in_open_grid() -> None:
    planner = TabularWorldModelPlanner(
        dynamics=_open_grid_dynamics,
        action_space=ACTIONS,
        num_candidates=300,
        plan_horizon=10,
        seed=0,
    )
    actions = planner.plan(observation=(0, 0), goal=(3, 3), horizon=10)
    pos: tuple[int, int] = (0, 0)
    for a in actions:
        pos = _open_grid_dynamics(pos, a)
    assert pos == (3, 3)


def test_plan_empty_when_horizon_zero() -> None:
    planner = TabularWorldModelPlanner(dynamics=_open_grid_dynamics, action_space=ACTIONS, seed=0)
    assert planner.plan(observation=(0, 0), goal=(1, 1), horizon=0) == []


def test_invalid_construction_args() -> None:
    with pytest.raises(ValueError):
        TabularWorldModelPlanner(dynamics=_open_grid_dynamics, action_space=ACTIONS, num_candidates=0)
    with pytest.raises(ValueError):
        TabularWorldModelPlanner(dynamics=_open_grid_dynamics, action_space=ACTIONS, plan_horizon=0)
    with pytest.raises(ValueError):
        TabularWorldModelPlanner(dynamics=_open_grid_dynamics, action_space=())
