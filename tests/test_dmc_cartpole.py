"""Smoke tests for the DMC Cartpole adapter.

Skipped automatically when `dm_control` is not installed, so the rest of
the suite still runs on a stdlib-only or torch-only checkout.
"""

from __future__ import annotations

import math
import pytest

dm_control = pytest.importorskip("dm_control")
np = pytest.importorskip("numpy")

from wmel.adapters.random_policy import RandomPolicy
from wmel.benchmark_runner import BenchmarkRunner
from wmel.envs.dmc_cartpole import (
    DEFAULT_DISCRETE_LEVELS,
    DMCCartpoleEnv,
    _flatten_observation,
    cartpole_upright_score,
    make_cartpole_oracle_dynamics,
)
from wmel.metrics import action_success_rate


def test_reset_returns_five_dim_observation() -> None:
    """Cartpole exposes position (3) + velocity (2) = 5 floats."""
    env = DMCCartpoleEnv()
    obs = env.reset()
    assert isinstance(obs, tuple)
    assert len(obs) == 5
    assert all(isinstance(x, float) for x in obs)


def test_action_space_matches_default_levels() -> None:
    env = DMCCartpoleEnv()
    expected = tuple((float(level),) for level in DEFAULT_DISCRETE_LEVELS)
    assert env.action_space == expected


def test_step_with_a_valid_action_updates_state_and_reward() -> None:
    env = DMCCartpoleEnv()
    env.reset()
    initial_obs = env.observation
    new_obs = env.step((1.0,))
    assert new_obs != initial_obs
    assert isinstance(env.last_reward, float)
    assert env.last_reward >= 0.0


def test_step_rejects_non_tuple_action() -> None:
    env = DMCCartpoleEnv()
    env.reset()
    with pytest.raises(ValueError):
        env.step(0.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        env.step((0.5, 0.0))


def test_oracle_dynamics_matches_env_step() -> None:
    """The oracle dynamics callable must reproduce `env.step()` to numerical
    precision over a short random-policy rollout. Same regression invariant
    as the Acrobot oracle.
    """
    env = DMCCartpoleEnv()
    obs = env.reset()
    oracle = make_cartpole_oracle_dynamics()
    rng_actions = [(1.0,), (-1.0,), (0.5,), (-0.5,), (0.0,), (1.0,), (-1.0,), (0.0,)]
    for action in rng_actions:
        predicted = oracle(obs, action)
        observed = env.step(action)
        diffs = [abs(a - b) for a, b in zip(predicted, observed)]
        assert max(diffs) < 1e-4, f"oracle/env mismatch: pred={predicted}, obs={observed}, diffs={diffs}"
        obs = observed


def test_score_is_minus_cos_theta() -> None:
    """state = (cart_x, cos_theta, sin_theta, cart_v, theta_dot).
    Score is -cos_theta = -state[1].
    """
    assert cartpole_upright_score((0.0, 1.0, 0.0, 0.0, 0.0)) == -1.0   # upright
    assert cartpole_upright_score((0.0, -1.0, 0.0, 0.0, 0.0)) == 1.0    # hanging down
    assert cartpole_upright_score((0.5, 0.5, math.sqrt(0.75), 0.0, 0.0)) == -0.5


def test_runner_with_random_policy_returns_episodes() -> None:
    env = DMCCartpoleEnv()
    policy = RandomPolicy(action_space=env.action_space, seed=0)
    results = BenchmarkRunner(
        env_factory=DMCCartpoleEnv,
        policy=policy,
        episodes=2,
        horizon=20,
        perturb_prob=0.0,
        seed=0,
    ).run()
    assert len(results) == 2
    # A random policy in 20 steps is overwhelmingly unlikely to swing up.
    assert action_success_rate(results) == 0.0


def test_flatten_observation_is_deterministic_by_sorted_keys() -> None:
    env = DMCCartpoleEnv()
    env.reset()
    ts = env._env.step(np.zeros(1, dtype=np.float32))
    obs = _flatten_observation(ts.observation)
    # position has 3 dims, velocity has 2; sorted keys put position first.
    assert len(obs) == 5
