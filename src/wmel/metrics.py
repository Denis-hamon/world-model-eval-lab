"""Decision-grade metrics for action-conditioned world model evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import fmean
from typing import Iterable, Sequence


@dataclass(frozen=True)
class EpisodeResult:
    """The result of one benchmark episode.

    Attributes
    ----------
    success
        Whether the agent reached the goal within the horizon.
    steps
        Number of environment steps taken in the episode.
    planning_latencies_ms
        Per-call wall-clock latency of every `policy.plan(...)` invocation
        during the episode. Stored as a tuple so the per-call distribution
        survives aggregation (the "Planning Latency" metric in the taxonomy
        is per-call, not per-episode).
    perturbed
        Whether `env.perturb()` was actually called during the episode.
        False for episodes that finished before the scheduled perturbation
        step, even when the runner had selected the episode for perturbation.
    recovered
        True iff `perturbed` is True and the agent still reached the goal.
    compute_per_decision
        Optional, model-reported compute cost per planned action (FLOPs,
        forward passes, or any agreed-upon unit). `None` means not measured.
    """

    success: bool
    steps: int
    planning_latencies_ms: tuple[float, ...] = ()
    perturbed: bool = False
    recovered: bool = False
    compute_per_decision: float | None = None

    @property
    def total_planning_latency_ms(self) -> float:
        return sum(self.planning_latencies_ms)

    @property
    def plan_calls(self) -> int:
        return len(self.planning_latencies_ms)


@dataclass(frozen=True)
class Scorecard:
    """A compact summary of a benchmark run, suitable for reporting."""

    policy_name: str
    episodes: int
    success_rate: float
    average_steps_to_success: float | None
    average_planning_latency_ms: float
    perturbation_recovery_rate: float | None
    average_compute_per_decision: float | None = None
    perturbation_name: str | None = None
    extras: dict[str, float] = field(default_factory=dict)


def action_success_rate(results: Sequence[EpisodeResult]) -> float:
    """Fraction of episodes that reached the goal."""
    if not results:
        return 0.0
    return sum(1 for r in results if r.success) / len(results)


def average_steps_to_success(results: Sequence[EpisodeResult]) -> float | None:
    """Mean number of steps over successful episodes. None if none succeeded."""
    successes = [r.steps for r in results if r.success]
    if not successes:
        return None
    return fmean(successes)


def average_planning_latency_ms(results: Sequence[EpisodeResult]) -> float:
    """Mean wall-clock latency per `policy.plan(...)` call, across all episodes.

    This is a per-call mean, not a per-episode mean. A policy that replans more
    often does not get a free pass: every plan() invocation contributes equally.
    """
    latencies = [lat for r in results for lat in r.planning_latencies_ms]
    if not latencies:
        return 0.0
    return fmean(latencies)


def perturbation_recovery_rate(results: Sequence[EpisodeResult]) -> float | None:
    """Among perturbed episodes, the fraction that still reached the goal.

    Returns None if no episodes were perturbed - the metric is meaningless
    without a denominator.
    """
    perturbed = [r for r in results if r.perturbed]
    if not perturbed:
        return None
    return sum(1 for r in perturbed if r.recovered) / len(perturbed)


def _average_compute_per_decision(
    results: Sequence[EpisodeResult],
    compute_per_plan_call: float | None,
) -> float | None:
    """Derive average compute per executed action.

    Precedence: when `compute_per_plan_call` is not None (including 0.0), the
    derivation `(compute_per_plan_call * total_plan_calls) / total_steps` wins
    and any per-episode `EpisodeResult.compute_per_decision` values are
    ignored. Pass `None` explicitly to fall back to per-episode telemetry.
    """
    if compute_per_plan_call is not None:
        total_plan_calls = sum(r.plan_calls for r in results)
        total_steps = sum(r.steps for r in results)
        if total_steps == 0:
            return None
        return (compute_per_plan_call * total_plan_calls) / total_steps

    measured = [r.compute_per_decision for r in results if r.compute_per_decision is not None]
    if not measured:
        return None
    return fmean(measured)


def compute_scorecard(
    results: Sequence[EpisodeResult],
    policy_name: str,
    extras: Iterable[tuple[str, float]] | None = None,
    compute_per_plan_call: float | None = None,
    perturbation_name: str | None = None,
) -> Scorecard:
    """Aggregate a list of `EpisodeResult` into a `Scorecard`.

    Pass `compute_per_plan_call` (typically `policy.compute_per_plan_call`)
    to populate the `average_compute_per_decision` field. When None, the
    scorecard reports `None` for that field.

    Pass `perturbation_name` (typically `runner.perturbation.name`) to record
    which perturbation strategy was used. Two scorecards with the same policy
    but different perturbations should be distinguishable in their JSON and
    rendered output.
    """
    return Scorecard(
        policy_name=policy_name,
        episodes=len(results),
        success_rate=action_success_rate(results),
        average_steps_to_success=average_steps_to_success(results),
        average_planning_latency_ms=average_planning_latency_ms(results),
        perturbation_recovery_rate=perturbation_recovery_rate(results),
        average_compute_per_decision=_average_compute_per_decision(results, compute_per_plan_call),
        perturbation_name=perturbation_name,
        extras=dict(extras) if extras else {},
    )
