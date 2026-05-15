"""Tests for the maze toy environment and the tabular world model planner."""

from __future__ import annotations

from wmel.adapters.greedy_policy import GreedyGridPolicy
from wmel.adapters.random_policy import RandomPolicy
from wmel.adapters.tabular_world_model import TabularWorldModelPlanner
from wmel.benchmark_runner import BenchmarkRunner
from wmel.metrics import action_success_rate

from examples.maze_toy.environment import VALID_ACTIONS, MazeEnv


def test_maze_resets_to_start() -> None:
    env = MazeEnv()
    obs = env.reset()
    assert obs == env.start
    assert env.observation == env.start
    assert env.action_space == VALID_ACTIONS
    assert not env.is_success()


def test_maze_walls_block_direct_path() -> None:
    env = MazeEnv()
    env.reset()
    # The default start is (1,5). Cells (2,5) and (0,5) are walls,
    # (1,6) is a wall border, so only "down" should change the position.
    before = env.observation
    env.step("right")
    assert env.observation == before
    env.step("left")
    assert env.observation == before
    env.step("up")
    assert env.observation == before
    env.step("down")
    assert env.observation != before


def test_maze_dynamics_matches_step() -> None:
    env = MazeEnv()
    env.reset()
    pos = env.observation
    for action in ("down", "down", "down", "right", "right"):
        predicted = env.dynamics(pos, action)
        actual = env.step(action)
        assert predicted == actual
        pos = actual


def test_tabular_world_model_beats_random_on_maze() -> None:
    template = MazeEnv()

    random_results = BenchmarkRunner(
        env_factory=MazeEnv,
        policy=RandomPolicy(action_space=VALID_ACTIONS, seed=0),
        episodes=10,
        horizon=80,
        perturb_prob=0.0,
        seed=0,
    ).run()

    wm_planner = TabularWorldModelPlanner(
        dynamics=template.dynamics,
        action_space=VALID_ACTIONS,
        num_candidates=200,
        plan_horizon=20,
        seed=0,
    )
    wm_results = BenchmarkRunner(
        env_factory=MazeEnv,
        policy=wm_planner,
        episodes=10,
        horizon=80,
        perturb_prob=0.0,
        seed=0,
    ).run()

    assert action_success_rate(wm_results) > action_success_rate(random_results)
    assert action_success_rate(wm_results) >= 0.5


def test_naive_greedy_gets_stuck_on_maze() -> None:
    """Naive Manhattan greedy with no waypoint cannot solve the maze.

    This documents the limitation that motivates the tabular world model.
    """
    results = BenchmarkRunner(
        env_factory=MazeEnv,
        policy=GreedyGridPolicy(),
        episodes=5,
        horizon=80,
        perturb_prob=0.0,
        seed=0,
    ).run()
    assert action_success_rate(results) == 0.0
