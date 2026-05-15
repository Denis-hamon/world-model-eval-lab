"""Tests for the experiments module: CI helpers and horizon_sweep."""

from __future__ import annotations

import pytest

from wmel.adapters.tabular_world_model import TabularWorldModelPlanner
from wmel.experiments import (
    horizon_sweep,
    mean_normal_interval,
    wilson_interval,
)

from examples.maze_toy.environment import VALID_ACTIONS, MazeEnv


def test_wilson_interval_bounds() -> None:
    low, high = wilson_interval(5, 10)
    assert 0.0 <= low <= 0.5 <= high <= 1.0


def test_wilson_interval_all_success() -> None:
    low, high = wilson_interval(10, 10)
    assert low > 0.5
    assert high == 1.0


def test_wilson_interval_zero_total() -> None:
    assert wilson_interval(0, 0) == (0.0, 0.0)


def test_mean_normal_interval_centered_on_mean() -> None:
    values = [1.0, 1.0, 1.0, 1.0, 1.0]
    low, high = mean_normal_interval(values)
    assert low == pytest.approx(1.0)
    assert high == pytest.approx(1.0)


def test_mean_normal_interval_single_value() -> None:
    low, high = mean_normal_interval([3.0])
    assert low == 3.0
    assert high == 3.0


def test_mean_normal_interval_empty() -> None:
    assert mean_normal_interval([]) == (0.0, 0.0)


def test_horizon_sweep_produces_one_point_per_horizon() -> None:
    template = MazeEnv()

    def factory(plan_horizon: int) -> TabularWorldModelPlanner:
        return TabularWorldModelPlanner(
            dynamics=template.dynamics,
            action_space=VALID_ACTIONS,
            num_candidates=50,
            plan_horizon=plan_horizon,
            seed=0,
        )

    sweep = horizon_sweep(
        env_factory=MazeEnv,
        policy_factory=factory,
        plan_horizons=(5, 15),
        episodes_per_point=4,
        episode_horizon=60,
        seed=0,
    )
    assert len(sweep.points) == 2
    assert sweep.points[0].plan_horizon == 5
    assert sweep.points[1].plan_horizon == 15
    assert sweep.policy_name == "tabular-world-model"


def test_horizon_sweep_long_horizon_beats_short_horizon() -> None:
    """A longer planning horizon should not hurt success rate on the maze."""
    template = MazeEnv()

    def factory(plan_horizon: int) -> TabularWorldModelPlanner:
        return TabularWorldModelPlanner(
            dynamics=template.dynamics,
            action_space=VALID_ACTIONS,
            num_candidates=200,
            plan_horizon=plan_horizon,
            seed=0,
        )

    sweep = horizon_sweep(
        env_factory=MazeEnv,
        policy_factory=factory,
        plan_horizons=(5, 20),
        episodes_per_point=10,
        episode_horizon=80,
        seed=0,
    )
    short = sweep.points[0].scorecard.success_rate
    long_ = sweep.points[1].scorecard.success_rate
    assert long_ > short


def test_horizon_sweep_rejects_empty_horizons() -> None:
    with pytest.raises(ValueError):
        horizon_sweep(
            env_factory=MazeEnv,
            policy_factory=lambda h: None,  # type: ignore[arg-type, return-value]
            plan_horizons=(),
        )


def test_horizon_sweep_forwards_perturbation_kwarg_to_runner() -> None:
    """v0.7 contract: horizon_sweep accepts a Perturbation; episodes get the
    custom strategy via BenchmarkRunner, not the EnvPerturbation default."""
    from wmel.adapters.base import BenchmarkEnvironment, PlannerPolicy
    from wmel.perturbations import Perturbation

    fired: list[int] = []

    class _NoOpEnv(BenchmarkEnvironment):
        def reset(self): return 0
        def step(self, a): return 0
        def is_success(self): return False
        def perturb(self): pass

        @property
        def observation(self): return 0
        @property
        def goal(self): return 1
        @property
        def action_space(self): return ("up",)

    class _OneStepPolicy(PlannerPolicy):
        @property
        def name(self): return "one-step"
        def plan(self, observation, goal, horizon): return ["up"] if horizon > 0 else []

    class _CountingPerturbation(Perturbation):
        @property
        def name(self): return "counting"

        def apply_to_env(self, env):
            fired.append(1)

    sweep = horizon_sweep(
        env_factory=_NoOpEnv,
        policy_factory=lambda h: _OneStepPolicy(),
        plan_horizons=(2,),
        episodes_per_point=10,
        episode_horizon=4,
        perturb_prob=1.0,
        perturbation=_CountingPerturbation(),
        seed=0,
    )
    assert sweep.points[0].plan_horizon == 2
    # Each of the 10 episodes was selected for perturbation and gets to the
    # scheduled step before the horizon runs out, so the custom Perturbation
    # fires - not the env default.
    assert len(fired) >= 1


def test_horizon_sweep_default_perturbation_unchanged() -> None:
    """Backward-compat: omitting `perturbation=` keeps pre-v0.7 behaviour."""
    template = MazeEnv()

    def factory(plan_horizon: int) -> TabularWorldModelPlanner:
        return TabularWorldModelPlanner(
            dynamics=template.dynamics,
            action_space=VALID_ACTIONS,
            num_candidates=50,
            plan_horizon=plan_horizon,
            seed=0,
        )

    sweep = horizon_sweep(
        env_factory=MazeEnv,
        policy_factory=factory,
        plan_horizons=(15,),
        episodes_per_point=4,
        episode_horizon=60,
        perturb_prob=0.0,
        seed=0,
    )
    assert sweep.points[0].plan_horizon == 15
    assert sweep.points[0].scorecard.episodes == 4
