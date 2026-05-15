"""Tests for the Markdown reporting helpers."""

from __future__ import annotations

import re

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
    # None-valued metrics render as n/a. Both perturbation recovery rate and
    # compute-per-decision are None for this fixture, so both rows show n/a.
    assert "| perturbation recovery rate | n/a |" in md
    assert "| average compute per decision | n/a |" in md


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
    # Compute-per-decision must appear in the sweep table - it is the metric
    # whose plumbing v0.4 introduced, and the trade-off documented in 02 is
    # latency + horizon + compute reported together.
    assert "compute_per_decision" in md
    # Two data rows: any line whose first cell starts with a digit.
    data_row = re.compile(r"^\|\s*\d")
    rows = [line for line in md.strip().splitlines() if data_row.match(line)]
    assert len(rows) == 2


def test_scorecard_reports_compute_per_decision_when_policy_provides_it() -> None:
    """A `TabularWorldModelPlanner` advertises `compute_per_plan_call`.

    The scorecard derives `average_compute_per_decision` as
    (compute_per_plan_call * total_plan_calls) / total_steps.

    Fixture: 2 episodes, plan_calls = 2 + 1 = 3, steps = 10 + 20 = 30,
    compute_per_plan_call = 4000.
    Expected: (4000 * 3) / 30 = 400.0.
    """
    results = _two_synthetic_episodes()
    sc = compute_scorecard(
        results,
        policy_name="unit-policy",
        compute_per_plan_call=4000.0,
    )
    assert sc.average_compute_per_decision is not None
    assert abs(sc.average_compute_per_decision - 400.0) < 1e-9


def test_scorecard_reports_none_compute_when_not_provided() -> None:
    sc = compute_scorecard(_two_synthetic_episodes(), policy_name="unit-policy")
    assert sc.average_compute_per_decision is None
