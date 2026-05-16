"""Smoke tests for the DMC Acrobot adapter.

Skipped automatically when `dm_control` is not installed, so the rest of
the suite still runs on a stdlib-only or torch-only checkout.
"""

from __future__ import annotations

import pytest

dm_control = pytest.importorskip("dm_control")
np = pytest.importorskip("numpy")

from wmel.adapters.random_policy import RandomPolicy
from wmel.benchmark_runner import BenchmarkRunner
from wmel.envs.dmc_acrobot import (
    DEFAULT_DISCRETE_LEVELS,
    DMCAcrobotEnv,
    _flatten_observation,
)
from wmel.metrics import action_success_rate


def test_reset_returns_six_dim_observation() -> None:
    """Acrobot exposes orientations (4) + velocity (2) = 6 floats."""
    env = DMCAcrobotEnv()
    obs = env.reset()
    assert isinstance(obs, tuple)
    assert len(obs) == 6
    assert all(isinstance(x, float) for x in obs)


def test_action_space_matches_default_levels() -> None:
    env = DMCAcrobotEnv()
    expected = tuple((float(level),) for level in DEFAULT_DISCRETE_LEVELS)
    assert env.action_space == expected


def test_step_with_a_valid_action_updates_state_and_reward() -> None:
    env = DMCAcrobotEnv()
    env.reset()
    initial_obs = env.observation
    new_obs = env.step((1.0,))
    assert new_obs != initial_obs
    assert isinstance(env.last_reward, float)
    # Reward at rest is tiny but non-negative; never negative for Acrobot-swingup.
    assert env.last_reward >= 0.0


def test_step_rejects_non_tuple_action() -> None:
    env = DMCAcrobotEnv()
    env.reset()
    with pytest.raises(ValueError):
        env.step(0.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        env.step((0.5, 0.0))


def test_is_success_threshold_strictness() -> None:
    """Reset state is far from upright; is_success must be False."""
    env = DMCAcrobotEnv(upright_threshold=0.6)
    env.reset()
    assert env.is_success() is False
    # Step once with neutral torque; still far from upright.
    env.step((0.0,))
    assert env.is_success() is False


def test_flatten_observation_is_deterministic() -> None:
    """Same dict, same flattened tuple (sorted-keys invariant)."""
    obs = {
        "orientations": np.array([1.0, 2.0, 3.0, 4.0]),
        "velocity": np.array([5.0, 6.0]),
    }
    out1 = _flatten_observation(obs)
    out2 = _flatten_observation(obs)
    assert out1 == out2
    assert out1 == (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)


def test_runner_drives_acrobot_with_random_policy() -> None:
    """Two episodes of random control. Random policy should not solve the
    swing-up; the test pins the framework + adapter end-to-end, not the
    policy's success."""
    policy = RandomPolicy(action_space=DMCAcrobotEnv().action_space, seed=0)
    results = BenchmarkRunner(
        env_factory=DMCAcrobotEnv,
        policy=policy,
        episodes=2,
        horizon=100,
        perturb_prob=0.0,
        seed=0,
    ).run()
    assert len(results) == 2
    # Random + a 100-step horizon is nowhere near enough to swing up.
    assert action_success_rate(results) == 0.0
    # Each episode either reached the horizon, or stopped earlier; both
    # are consistent with the runner's contract.
    assert all(r.steps <= 100 for r in results)
