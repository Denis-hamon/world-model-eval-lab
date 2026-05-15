"""Tests for the PyTorch-learned dynamics adapter.

Skipped automatically when torch is not installed (see `pytest.importorskip`),
so the rest of the suite keeps running on a stdlib-only checkout.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from wmel.adapters.learned_dynamics_torch import (
    MazeDynamicsMLP,
    collect_transitions,
    torch_dynamics,
    train_maze_dynamics,
)
from wmel.adapters.tabular_world_model import TabularWorldModelPlanner
from wmel.benchmark_runner import BenchmarkRunner
from wmel.metrics import action_success_rate

from examples.maze_toy.environment import VALID_ACTIONS, MazeEnv


def test_collect_transitions_covers_every_free_cell_action_pair() -> None:
    """The maze has 16 free cells; with 4 actions, 64 transitions enumerated."""
    env = MazeEnv()
    transitions = collect_transitions(env)
    assert len(transitions) == 16 * 4

    states = {s for s, _, _ in transitions}
    assert len(states) == 16
    actions = {a for _, a, _ in transitions}
    assert actions == set(VALID_ACTIONS)


def test_collect_transitions_rejects_envs_without_dynamics() -> None:
    """Missing-contract envs must fail loudly, not silently produce empty data."""

    class _BadEnv:
        width = 7
        height = 7
        walls: set = set()
        # no .dynamics method

    with pytest.raises(TypeError):
        collect_transitions(_BadEnv())  # type: ignore[arg-type]


def test_trained_mlp_recovers_oracle_dynamics_exactly() -> None:
    """800 epochs on 64 transitions should memorise the maze with zero errors.

    This is the core proof-of-contract: the learned model's output matches
    the env's hand-written dynamics on every reachable transition.
    """
    env = MazeEnv()
    model = train_maze_dynamics(env, epochs=800, seed=0)
    learned = torch_dynamics(model, env.width, env.height)

    transitions = collect_transitions(env)
    matches = sum(1 for s, a, ns in transitions if learned(s, a) == ns)
    assert matches == len(transitions), (
        f"learned dynamics matches oracle on only {matches}/{len(transitions)} "
        "transitions; the MLP is undertrained or the encoding is broken"
    )


def test_mlp_dynamics_drives_planner_to_success() -> None:
    """End-to-end: TabularWorldModelPlanner with learned dynamics solves the maze."""
    env = MazeEnv()
    model = train_maze_dynamics(env, epochs=800, seed=0)
    learned = torch_dynamics(model, env.width, env.height)

    planner = TabularWorldModelPlanner(
        dynamics=learned,
        action_space=VALID_ACTIONS,
        num_candidates=200,
        plan_horizon=20,
        seed=0,
    )
    results = BenchmarkRunner(
        env_factory=MazeEnv,
        policy=planner,
        episodes=10,
        horizon=80,
        perturb_prob=0.0,
        seed=0,
    ).run()
    assert action_success_rate(results) == 1.0


def test_mlp_forward_shape() -> None:
    """Smoke check on the MLP itself, independent of the maze."""
    model = MazeDynamicsMLP(hidden=16)
    out = model(torch.zeros(3, 6))
    assert out.shape == (3, 2)
