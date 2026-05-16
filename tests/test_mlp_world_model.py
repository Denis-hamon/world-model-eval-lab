"""Tests for the Markovian MLP world model adapter.

Skipped automatically when torch or dm_control is not installed, so the rest
of the suite still runs on stdlib-only and torch-only checkouts.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
dm_control = pytest.importorskip("dm_control")

from wmel.adapters.mlp_world_model import (
    MLPWorldModel,
    acrobot_upright_score,
    collect_random_rollouts,
    learned_dynamics,
    train_world_model,
)
from wmel.adapters.random_policy import RandomPolicy
from wmel.adapters.tabular_world_model import TabularWorldModelPlanner
from wmel.benchmark_runner import BenchmarkRunner
from wmel.envs.dmc_acrobot import DMCAcrobotEnv
from wmel.metrics import action_success_rate


def test_mlp_forward_shape() -> None:
    """Smoke check on the architecture, independent of any env."""
    model = MLPWorldModel(obs_dim=6, n_actions=5, hidden=16)
    obs = torch.zeros(4, 6)
    act = torch.tensor([0, 1, 2, 3])
    out = model(obs, act)
    assert out.shape == (4, 6)


def test_mlp_rejects_invalid_construction() -> None:
    with pytest.raises(ValueError):
        MLPWorldModel(obs_dim=0, n_actions=5)
    with pytest.raises(ValueError):
        MLPWorldModel(obs_dim=6, n_actions=0)
    with pytest.raises(ValueError):
        MLPWorldModel(obs_dim=6, n_actions=5, hidden=0)


def test_collect_random_rollouts_produces_transitions() -> None:
    transitions = collect_random_rollouts(
        DMCAcrobotEnv,
        n_episodes=2,
        max_steps_per_episode=10,
        seed=0,
    )
    # 2 episodes x 10 steps = 20 transitions (Acrobot never succeeds in 10 random steps).
    assert len(transitions) == 20
    obs0, a0, next0 = transitions[0]
    assert isinstance(obs0, tuple) and len(obs0) == 6
    assert isinstance(a0, int) and 0 <= a0 < 5
    assert isinstance(next0, tuple) and len(next0) == 6


def test_training_drives_val_mse_below_threshold() -> None:
    """Smoke check that the optimiser actually optimises.

    With 5 episodes of 100 steps each and 50 epochs, the val MSE should
    fall well below the starting reset variance (~10) on a 6-d state. We
    pin a relaxed threshold (0.5) to avoid flakiness; locally we see
    ~0.02 with the same seed.
    """
    transitions = collect_random_rollouts(
        DMCAcrobotEnv,
        n_episodes=5,
        max_steps_per_episode=100,
        seed=0,
    )
    model, log = train_world_model(
        transitions,
        obs_dim=6,
        n_actions=5,
        epochs=50,
        seed=0,
    )
    assert log["final_val_mse"] < 0.5
    assert log["final_train_mse"] < 0.5
    # Training should actually reduce the loss, not stay near init.
    assert log["final_train_mse"] < 1.0


def test_learned_dynamics_returns_a_tuple_of_floats() -> None:
    model = MLPWorldModel(obs_dim=6, n_actions=5, hidden=8)
    env = DMCAcrobotEnv()
    dyn = learned_dynamics(model, env.action_space)
    obs = env.reset()
    next_obs = dyn(obs, env.action_space[0])
    assert isinstance(next_obs, tuple)
    assert len(next_obs) == 6
    assert all(isinstance(x, float) for x in next_obs)


def test_acrobot_upright_score_lower_when_higher_tip() -> None:
    """Hand-crafted vectors that satisfy the DMC layout `(sin_u, sin_l, cos_u, cos_l, v0, v1)`
    and the unit-norm invariant `sin_x^2 + cos_x^2 = 1` for x in {upper, lower}.

    Higher tip = lower score (planner minimises).
    """
    # Both arms straight up: angles = 0, so (sin, cos) = (0, 1).
    up = (0.0, 0.0, 1.0, 1.0, 0.0, 0.0)
    # Both arms horizontal: angles = pi/2, so (sin, cos) = (1, 0).
    horizontal = (1.0, 1.0, 0.0, 0.0, 0.0, 0.0)
    # Both arms hanging straight down: angles = pi, so (sin, cos) = (0, -1).
    down = (0.0, 0.0, -1.0, -1.0, 0.0, 0.0)
    s_up = acrobot_upright_score(up)
    s_horiz = acrobot_upright_score(horizontal)
    s_down = acrobot_upright_score(down)
    # Up should be the lowest (best), down the highest (worst).
    assert s_up < s_horiz < s_down
    # Numeric sanity: -tip_y, so up = -2, horizontal = 0, down = +2.
    assert abs(s_up - (-2.0)) < 1e-9
    assert abs(s_horiz - 0.0) < 1e-9
    assert abs(s_down - 2.0) < 1e-9


def test_acrobot_upright_score_matches_dmc_layout() -> None:
    """Sanity-check that the score works on an actual DMC observation, not just
    on hand-crafted vectors. Pins the layout assumption against drift in
    dm-control."""
    env = DMCAcrobotEnv()
    obs = env.reset()
    # Unit-norm invariant on (sin_u, cos_u) and (sin_l, cos_l).
    assert abs(obs[0] ** 2 + obs[2] ** 2 - 1.0) < 1e-3
    assert abs(obs[1] ** 2 + obs[3] ** 2 - 1.0) < 1e-3
    # The score is a finite float for any valid observation.
    score = acrobot_upright_score(obs)
    assert isinstance(score, float)
    # Worst-case bounds: tip_y is in [-2, 2], so score is in [-2, 2].
    assert -2.0 - 1e-6 <= score <= 2.0 + 1e-6


def test_planner_with_learned_dynamics_runs_end_to_end() -> None:
    """Pin the contract end-to-end. Success rate is NOT asserted: random
    rollouts do not cover the upright region, so we honestly expect 0%
    success - the distribution-shift problem the CPG metric (v1.0) will
    quantify."""
    env_template = DMCAcrobotEnv()
    transitions = collect_random_rollouts(
        DMCAcrobotEnv,
        n_episodes=5,
        max_steps_per_episode=80,
        seed=0,
    )
    model, _ = train_world_model(
        transitions,
        obs_dim=6,
        n_actions=5,
        epochs=30,
        seed=0,
    )
    dyn = learned_dynamics(model, env_template.action_space)
    planner = TabularWorldModelPlanner(
        dynamics=dyn,
        action_space=env_template.action_space,
        num_candidates=20,
        plan_horizon=10,
        score=acrobot_upright_score,
        seed=0,
    )
    results = BenchmarkRunner(
        env_factory=DMCAcrobotEnv,
        policy=planner,
        episodes=2,
        horizon=80,
        perturb_prob=0.0,
        seed=0,
    ).run()
    assert len(results) == 2
    # Each episode took some steps (could be horizon-capped at 80 or success-capped).
    assert all(r.steps > 0 for r in results)
    # Success rate is honestly bounded by the random-rollout coverage of
    # the upright region: we do not assert a floor, only that the run
    # completes and produces a valid scorecard structure.
    sr = action_success_rate(results)
    assert 0.0 <= sr <= 1.0
