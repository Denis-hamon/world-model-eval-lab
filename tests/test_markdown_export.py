"""Tests for the Markdown reporting helpers."""

from __future__ import annotations

from wmel.adapters.tabular_world_model import TabularWorldModelPlanner
from wmel.experiments import horizon_sweep, to_markdown_horizon_sweep
from wmel.metrics import EpisodeResult, compute_scorecard
from wmel.report import to_markdown_report, to_markdown_scorecard

from examples.maze_toy.environment import VALID_ACTIONS, MazeEnv


def _two_synthetic_episodes() -> list[EpisodeResult]:
    return [
        EpisodeResult(success=True, steps=10, planning_latencies_ms=(1.0, 2.0)),
        EpisodeResult(success=False, steps=20, planning_latencies_ms=(3.0,)),
    ]


def test_markdown_scorecard_is_valid_markdown_table() -> None:
    sc = compute_scorecard(_two_synthetic_episodes(), policy_name="unit-policy")
    md = to_markdown_scorecard(sc)
    assert "### Scorecard: `unit-policy`" in md
    assert "| Metric | Value |" in md
    assert "| --- | --- |" in md
    assert "| episodes | 2 |" in md
    assert "action success rate" in md
    assert "perturbation recovery rate" in md
    # The CI list should be n/a when there is no value.
    assert "n/a" in md


def test_markdown_report_includes_each_scorecard() -> None:
    sc_a = compute_scorecard(_two_synthetic_episodes(), policy_name="alpha")
    sc_b = compute_scorecard(_two_synthetic_episodes(), policy_name="beta")
    md = to_markdown_report([sc_a, sc_b], heading="Comparison")
    assert md.startswith("# Comparison")
    assert "`alpha`" in md
    assert "`beta`" in md


def test_markdown_horizon_sweep_renders_one_row_per_point() -> None:
    template = MazeEnv()

    def factory(plan_horizon: int) -> TabularWorldModelPlanner:
        return TabularWorldModelPlanner(
            dynamics=template.dynamics,
            action_space=VALID_ACTIONS,
            num_candidates=20,
            plan_horizon=plan_horizon,
            seed=0,
        )

    sweep = horizon_sweep(
        env_factory=MazeEnv,
        policy_factory=factory,
        plan_horizons=(5, 15),
        episodes_per_point=3,
        episode_horizon=40,
        seed=0,
    )
    md = to_markdown_horizon_sweep(sweep)
    assert md.startswith("### Horizon sweep: `tabular-world-model`")
    assert "| plan_horizon | success_rate" in md
    # Two data rows; the header occupies the first three lines.
    rows = [line for line in md.strip().splitlines() if line.startswith("|") and line[2].isdigit()]
    assert len(rows) == 2


def test_scorecard_reports_compute_per_decision_when_policy_provides_it() -> None:
    """A `TabularWorldModelPlanner` advertises `compute_per_plan_call`.

    The scorecard derives `average_compute_per_decision` as
    (compute_per_plan_call * total_plan_calls) / total_steps. With 2 episodes
    of 1 plan call each, 10 + 20 steps, and 4000 cost per plan, the expected
    average is (4000 * 2) / 30 ≈ 266.67.
    """
    results = _two_synthetic_episodes()  # plan_calls = 2 and 1, total = 3, steps = 30
    sc = compute_scorecard(
        results,
        policy_name="unit-policy",
        compute_per_plan_call=4000.0,
    )
    # total compute = 4000 * (2 + 1) = 12000, total steps = 30
    assert sc.average_compute_per_decision is not None
    assert abs(sc.average_compute_per_decision - 400.0) < 1e-9


def test_scorecard_reports_none_compute_when_not_provided() -> None:
    sc = compute_scorecard(_two_synthetic_episodes(), policy_name="unit-policy")
    assert sc.average_compute_per_decision is None
