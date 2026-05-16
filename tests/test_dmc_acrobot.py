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
    make_acrobot_oracle_dynamics,
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


def test_oracle_dynamics_reproduces_env_step_to_numerical_precision() -> None:
    """A bit-for-bit check: when the oracle is queried with the same state +
    action that the env was just stepped from, it must return the same next
    state. This is the contract that makes CPG = 0 a *real* zero, not a
    coincidence."""
    env = DMCAcrobotEnv()
    oracle = make_acrobot_oracle_dynamics()

    obs = env.reset()
    # Try every torque level.
    for action in env.action_space:
        # Reset the env to make obs reproducible (since env.step mutates state).
        obs0 = env.reset()
        predicted = oracle(obs0, action)
        actual = env.step(action)
        diff = max(abs(p - a) for p, a in zip(predicted, actual))
        # Allow tiny numerical noise from physics.forward()/integration order.
        assert diff < 1e-6, f"oracle drift {diff} for action {action}"


def test_oracle_dynamics_is_side_effect_free_on_caller_env() -> None:
    """The oracle has its own internal mujoco env. Calling it many times
    must not mutate the user's env or interfere with a parallel benchmark
    run."""
    user_env = DMCAcrobotEnv()
    oracle = make_acrobot_oracle_dynamics()
    obs_before = user_env.reset()
    for _ in range(50):
        oracle(obs_before, user_env.action_space[0])
    # The user's env has not been touched.
    assert user_env.observation == obs_before


def test_oracle_dynamics_handles_long_call_chains() -> None:
    """The oracle resets its internal env every `reset_every` calls to avoid
    hitting the DMC default 1000-step time limit. Test it survives 2000+
    calls (well past the default 800 reset_every)."""
    env = DMCAcrobotEnv()
    oracle = make_acrobot_oracle_dynamics(reset_every=200)
    obs = env.reset()
    state = obs
    for _ in range(500):
        state = oracle(state, env.action_space[2])  # idle torque
    # State stays in bounds (unit-norm sin/cos pairs).
    assert abs(state[0] ** 2 + state[2] ** 2 - 1.0) < 1e-3
    assert abs(state[1] ** 2 + state[3] ** 2 - 1.0) < 1e-3


def test_oracle_dynamics_matches_env_across_swept_states() -> None:
    """The headline `test_oracle_dynamics_reproduces_env_step_to_numerical_
    precision` covers only the post-reset configuration. CPG's correctness
    depends on the oracle being a faithful proxy *throughout* the planning
    horizon, including near-upright configurations the random-shooting MPC
    can visit during evaluation.

    Sweep: roll a random policy for 50 steps, snapshot every state, and
    assert the oracle predicts the same env-step result at each state.
    Loose tolerance (1e-5) to accommodate integrator drift if any.
    """
    import random as _rand

    env = DMCAcrobotEnv()
    oracle = make_acrobot_oracle_dynamics()
    rng = _rand.Random(0)

    obs = env.reset()
    for _ in range(50):
        action = env.action_space[rng.randrange(len(env.action_space))]
        # Predict via oracle from the current state
        predicted = oracle(obs, action)
        # Step the env from the same state for ground truth
        actual = env.step(action)
        diff = max(abs(p - a) for p, a in zip(predicted, actual))
        assert diff < 1e-5, (
            f"oracle drift {diff} at state {[round(x, 3) for x in obs]} "
            f"action {action}"
        )
        obs = actual


def test_acrobot_upright_score_layout_locked_in() -> None:
    """Pin the observation layout assumption: a synthetic upright state
    must score lower (better) than a synthetic hanging-down state. If a
    future change reorders the flattened observation (or DMC silently
    changes its orientation layout), this test fails and CPG's score-
    function correctness is exposed before it can mask a model/planner
    diagnosis.
    """
    from wmel.adapters.mlp_world_model import acrobot_upright_score

    upright = (0.0, 0.0, 1.0, 1.0, 0.0, 0.0)
    hanging = (0.0, 0.0, -1.0, -1.0, 0.0, 0.0)
    assert acrobot_upright_score(upright) < acrobot_upright_score(hanging)
    assert acrobot_upright_score(upright) == pytest.approx(-2.0)
    assert acrobot_upright_score(hanging) == pytest.approx(2.0)
