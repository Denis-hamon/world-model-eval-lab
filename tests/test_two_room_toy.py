"""Integration tests for the two-room toy environment and baselines."""

from __future__ import annotations

from wmel.adapters.greedy_policy import GreedyGridPolicy
from wmel.adapters.random_policy import RandomPolicy
from wmel.benchmark_runner import BenchmarkRunner
from wmel.metrics import action_success_rate

from examples.two_room_toy.environment import (
    VALID_ACTIONS,
    TwoRoomEnv,
    two_room_waypoint_for,
)


def test_env_resets_to_start() -> None:
    env = TwoRoomEnv()
    obs = env.reset()
    assert obs == env.start
    assert env.observation == env.start
    assert not env.is_success()


def test_env_wall_blocks_movement() -> None:
    env = TwoRoomEnv()
    env.reset()
    # The default start is in the left room; stepping right repeatedly should
    # be blocked at the wall column except through the doorway row.
    for _ in range(env.width):
        env.step("right")
    x, _ = env.observation
    assert x <= env.wall_x


def test_greedy_reaches_goal_with_fixed_seed() -> None:
    env = TwoRoomEnv()
    waypoint_fn = two_room_waypoint_for(env)
    policy = GreedyGridPolicy(waypoint_fn=waypoint_fn)
    results = BenchmarkRunner(
        env_factory=TwoRoomEnv,
        policy=policy,
        episodes=10,
        horizon=60,
        perturb_prob=0.0,
        seed=0,
    ).run()
    assert action_success_rate(results) == 1.0


def test_greedy_beats_random_on_success_rate() -> None:
    random_policy = RandomPolicy(action_space=VALID_ACTIONS, seed=0)
    random_results = BenchmarkRunner(
        env_factory=TwoRoomEnv,
        policy=random_policy,
        episodes=30,
        horizon=60,
        perturb_prob=0.0,
        seed=0,
    ).run()

    waypoint_fn = two_room_waypoint_for(TwoRoomEnv())
    greedy_policy = GreedyGridPolicy(waypoint_fn=waypoint_fn)
    greedy_results = BenchmarkRunner(
        env_factory=TwoRoomEnv,
        policy=greedy_policy,
        episodes=30,
        horizon=60,
        perturb_prob=0.0,
        seed=0,
    ).run()

    assert action_success_rate(greedy_results) > action_success_rate(random_results)
