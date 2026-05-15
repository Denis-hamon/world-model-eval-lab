"""Unit tests for the metrics module against synthetic episode results."""

from __future__ import annotations

import pytest

from wmel.metrics import (
    EpisodeResult,
    action_success_rate,
    average_planning_latency_ms,
    average_steps_to_success,
    compute_scorecard,
    perturbation_recovery_rate,
)


def _synthetic_results() -> list[EpisodeResult]:
    return [
        EpisodeResult(success=True, steps=10, planning_latencies_ms=(5.0,)),
        EpisodeResult(success=False, steps=50, planning_latencies_ms=(7.0,)),
        EpisodeResult(success=True, steps=14, planning_latencies_ms=(3.0,)),
        EpisodeResult(success=True, steps=12, planning_latencies_ms=(5.0,), perturbed=True, recovered=True),
        EpisodeResult(success=False, steps=50, planning_latencies_ms=(5.0,), perturbed=True, recovered=False),
    ]


def test_action_success_rate() -> None:
    assert action_success_rate(_synthetic_results()) == pytest.approx(3 / 5)


def test_action_success_rate_empty() -> None:
    assert action_success_rate([]) == 0.0


def test_average_steps_to_success() -> None:
    assert average_steps_to_success(_synthetic_results()) == pytest.approx(12.0)


def test_average_steps_to_success_no_success() -> None:
    results = [EpisodeResult(success=False, steps=50, planning_latencies_ms=(1.0,))]
    assert average_steps_to_success(results) is None


def test_average_planning_latency_ms() -> None:
    assert average_planning_latency_ms(_synthetic_results()) == pytest.approx(5.0)


def test_average_planning_latency_is_per_call_not_per_episode() -> None:
    """Regression: a slow single plan() must not be masked by many cheap plans.

    Episode A: 3 plan() calls of 1 ms each   -> 3 ms total, 1 ms per call.
    Episode B: 1 plan() call of 9 ms         -> 9 ms total, 9 ms per call.

    Per-episode mean of totals: (3 + 9) / 2 = 6 ms (wrong, what the old code did).
    Per-call mean:              (1+1+1+9) / 4 = 3 ms (right, what the doc defines).
    """
    results = [
        EpisodeResult(success=True, steps=3, planning_latencies_ms=(1.0, 1.0, 1.0)),
        EpisodeResult(success=True, steps=1, planning_latencies_ms=(9.0,)),
    ]
    assert average_planning_latency_ms(results) == pytest.approx(3.0)


def test_perturbation_recovery_rate() -> None:
    assert perturbation_recovery_rate(_synthetic_results()) == pytest.approx(0.5)


def test_perturbation_recovery_rate_none_when_no_perturbations() -> None:
    results = [EpisodeResult(success=True, steps=1, planning_latencies_ms=(1.0,))]
    assert perturbation_recovery_rate(results) is None


def test_compute_scorecard_aggregates_all_fields() -> None:
    card = compute_scorecard(_synthetic_results(), policy_name="unit")
    assert card.policy_name == "unit"
    assert card.episodes == 5
    assert card.success_rate == pytest.approx(3 / 5)
    assert card.average_steps_to_success == pytest.approx(12.0)
    assert card.average_planning_latency_ms == pytest.approx(5.0)
    assert card.perturbation_recovery_rate == pytest.approx(0.5)
    assert card.average_compute_per_decision is None
    assert card.extras == {}
