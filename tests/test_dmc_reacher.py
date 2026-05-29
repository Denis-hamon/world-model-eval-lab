"""Smoke tests for the DMC Reacher adapter (the first 2-D-action env).

Skipped automatically when `dm_control` is not installed, so the rest of
the suite still runs on a stdlib-only or torch-only checkout.
"""

from __future__ import annotations

import pytest

dm_control = pytest.importorskip("dm_control")
np = pytest.importorskip("numpy")

from wmel.adapters.random_policy import RandomPolicy
from wmel.benchmark_runner import BenchmarkRunner
from wmel.envs.dmc_reacher import (
    DEFAULT_PER_DIM_LEVELS,
    DMCReacherEnv,
    _discrete_action_space,
    _flatten_observation,
    make_reacher_oracle_dynamics,
    reacher_reach_score,
)
from wmel.metrics import action_success_rate


def test_reset_returns_six_dim_observation() -> None:
    """Reacher exposes position (2) + to_target (2) + velocity (2) = 6 floats."""
    env = DMCReacherEnv()
    obs = env.reset()
    assert isinstance(obs, tuple)
    assert len(obs) == 6
    assert all(isinstance(x, float) for x in obs)


def test_action_space_is_cartesian_product_of_levels() -> None:
    env = DMCReacherEnv()
    # 3 levels per joint, 2 joints -> 9 actions, each a 2-tuple.
    assert len(env.action_space) == len(DEFAULT_PER_DIM_LEVELS) ** 2
    assert all(isinstance(a, tuple) and len(a) == 2 for a in env.action_space)
    assert env.action_space == _discrete_action_space(DEFAULT_PER_DIM_LEVELS)


def test_step_rejects_non_2tuple_action() -> None:
    env = DMCReacherEnv()
    env.reset()
    with pytest.raises(ValueError):
        env.step((0.5,))  # 1-tuple is wrong for a 2-D action env


def test_oracle_dynamics_matches_env_step() -> None:
    """The oracle must reproduce `env.step()` to numerical precision over a
    random-policy rollout. Reacher reconstruction is exact (position()=qpos,
    velocity()=qvel, target recovered from to_target), so the tolerance is
    tighter than the swing-up oracles' 1e-4.
    """
    env = DMCReacherEnv()
    obs = env.reset()
    oracle = make_reacher_oracle_dynamics()
    actions = [(1.0, -1.0), (-1.0, 1.0), (0.0, 1.0), (1.0, 0.0),
               (-1.0, -1.0), (0.0, 0.0), (1.0, 1.0), (0.0, -1.0)]
    for action in actions:
        predicted = oracle(obs, action)
        observed = env.step(action)
        diffs = [abs(a - b) for a, b in zip(predicted, observed)]
        assert max(diffs) < 1e-5, f"oracle/env mismatch: diffs={diffs}"
        obs = observed


def test_oracle_dynamics_matches_env_step_on_a_different_target() -> None:
    """Same oracle regression on a different episode target (random=7), so the
    target-recovery path (target_xy = finger_xy + to_target) is exercised on a
    target the oracle's own reset never saw. Guards against the oracle silently
    reusing a stale target geom position.
    """
    env = DMCReacherEnv(task_kwargs={"random": 7})
    obs = env.reset()
    oracle = make_reacher_oracle_dynamics(task_kwargs={"random": 0})
    actions = [(1.0, 1.0), (-1.0, 0.0), (0.0, -1.0), (1.0, -1.0), (-1.0, 1.0)]
    for action in actions:
        predicted = oracle(obs, action)
        observed = env.step(action)
        diffs = [abs(a - b) for a, b in zip(predicted, observed)]
        assert max(diffs) < 1e-5, f"oracle/env mismatch on random=7 target: diffs={diffs}"
        obs = observed


def test_score_is_finger_to_target_distance() -> None:
    """state = (pos0, pos1, to_target0, to_target1, vel0, vel1).
    Score is the Euclidean norm of to_target = sqrt(state[2]^2 + state[3]^2).
    """
    assert reacher_reach_score((0.0, 0.0, 0.0, 0.0, 0.0, 0.0)) == 0.0
    assert reacher_reach_score((0.0, 0.0, 3.0, 4.0, 0.0, 0.0)) == pytest.approx(5.0)


def test_runner_with_random_policy_returns_episodes() -> None:
    env = DMCReacherEnv()
    policy = RandomPolicy(action_space=env.action_space, seed=0)
    results = BenchmarkRunner(
        env_factory=DMCReacherEnv,
        policy=policy,
        episodes=2,
        horizon=20,
        perturb_prob=0.0,
        seed=0,
    ).run()
    assert len(results) == 2


def test_flatten_observation_is_deterministic_by_sorted_keys() -> None:
    env = DMCReacherEnv()
    env.reset()
    ts = env._env.step(np.zeros(2, dtype=np.float32))
    obs = _flatten_observation(ts.observation)
    # position(2) + to_target(2) + velocity(2), sorted keys -> 6 floats.
    assert len(obs) == 6
