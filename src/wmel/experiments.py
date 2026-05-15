"""Higher-level experiment helpers: horizon sweeps and confidence intervals.

The first useful experiment built on top of `BenchmarkRunner` is the
**planning-horizon sweep**: run the same policy on the same environment at
several lookahead depths and report how performance scales with horizon.
This is the experimental basis for the "Planning Horizon" metric documented
in `docs/02_metric_taxonomy.md`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from statistics import fmean, pstdev
from typing import Callable, Sequence

from wmel.adapters.base import BenchmarkEnvironment, PlannerPolicy
from wmel.benchmark_runner import BenchmarkRunner
from wmel.metrics import EpisodeResult, Scorecard, compute_scorecard


EnvFactory = Callable[[], BenchmarkEnvironment]
PolicyFactory = Callable[[int], PlannerPolicy]


@dataclass(frozen=True)
class HorizonSweepPoint:
    """One point on a planning-horizon sweep curve."""

    plan_horizon: int
    scorecard: Scorecard
    success_ci_low: float
    success_ci_high: float
    latency_ci_low: float
    latency_ci_high: float


@dataclass(frozen=True)
class HorizonSweep:
    """A full planning-horizon sweep, ready to print or to serialize."""

    policy_name: str
    points: list[HorizonSweepPoint] = field(default_factory=list)


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    More reliable than the normal approximation when the success rate is
    close to 0 or 1, which is precisely where horizon sweeps spend most of
    their data.
    """
    if total <= 0:
        return (0.0, 0.0)
    p = successes / total
    n = total
    denom = 1.0 + (z * z) / n
    center = (p + (z * z) / (2.0 * n)) / denom
    half = (z * math.sqrt((p * (1.0 - p) / n) + (z * z) / (4.0 * n * n))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def mean_normal_interval(
    values: Sequence[float],
    z: float = 1.96,
) -> tuple[float, float]:
    """Symmetric normal CI on the sample mean. Falls back to (mean, mean) for n<2."""
    n = len(values)
    if n == 0:
        return (0.0, 0.0)
    mean = fmean(values)
    if n < 2:
        return (mean, mean)
    sd = pstdev(values)
    se = sd / math.sqrt(n)
    return (mean - z * se, mean + z * se)


def horizon_sweep(
    env_factory: EnvFactory,
    policy_factory: PolicyFactory,
    plan_horizons: Sequence[int],
    episodes_per_point: int = 50,
    episode_horizon: int = 80,
    perturb_prob: float = 0.0,
    seed: int = 0,
) -> HorizonSweep:
    """Run `policy_factory(h)` for each `h` in `plan_horizons` and aggregate."""
    if not plan_horizons:
        raise ValueError("plan_horizons must not be empty")

    points: list[HorizonSweepPoint] = []
    policy_name: str | None = None

    for h in plan_horizons:
        policy = policy_factory(h)
        if policy_name is None:
            policy_name = policy.name
        results = BenchmarkRunner(
            env_factory=env_factory,
            policy=policy,
            episodes=episodes_per_point,
            horizon=episode_horizon,
            perturb_prob=perturb_prob,
            seed=seed,
        ).run()
        points.append(
            _summarize(
                h,
                results,
                policy_name=policy.name,
                compute_per_plan_call=policy.compute_per_plan_call,
            )
        )

    return HorizonSweep(policy_name=policy_name or "unknown", points=points)


def _summarize(
    plan_horizon: int,
    results: Sequence[EpisodeResult],
    policy_name: str,
    compute_per_plan_call: float | None = None,
) -> HorizonSweepPoint:
    scorecard = compute_scorecard(
        results,
        policy_name=policy_name,
        compute_per_plan_call=compute_per_plan_call,
    )
    successes = sum(1 for r in results if r.success)
    s_low, s_high = wilson_interval(successes, len(results))
    per_call_latencies = [lat for r in results for lat in r.planning_latencies_ms]
    l_low, l_high = mean_normal_interval(per_call_latencies)
    return HorizonSweepPoint(
        plan_horizon=plan_horizon,
        scorecard=scorecard,
        success_ci_low=s_low,
        success_ci_high=s_high,
        latency_ci_low=l_low,
        latency_ci_high=l_high,
    )


def to_markdown_horizon_sweep(sweep: HorizonSweep) -> str:
    """Render a `HorizonSweep` as a Markdown table, paste-ready for a doc."""
    lines = [
        f"### Horizon sweep: `{sweep.policy_name}`",
        "",
        "| plan_horizon | success_rate | success_95ci | avg_steps | latency_ms_per_call | latency_95ci |",
        "| ---: | ---: | :--- | ---: | ---: | :--- |",
    ]
    for point in sweep.points:
        sc = point.scorecard
        steps = "n/a" if sc.average_steps_to_success is None else f"{sc.average_steps_to_success:.1f}"
        lines.append(
            f"| {point.plan_horizon} | {sc.success_rate:.3f} | "
            f"[{point.success_ci_low:.2f}, {point.success_ci_high:.2f}] | "
            f"{steps} | {sc.average_planning_latency_ms:.3f} | "
            f"[{point.latency_ci_low:.2f}, {point.latency_ci_high:.2f}] |"
        )
    return "\n".join(lines) + "\n"


def print_horizon_sweep(sweep: HorizonSweep) -> None:
    """Print a compact ASCII table of a horizon sweep."""
    print(f"Horizon sweep: {sweep.policy_name}")
    header = (
        f"  {'plan_h':>6} | "
        f"{'success':>9} | {'95% CI':>15} | "
        f"{'steps':>7} | {'latency_ms':>10} | {'95% CI (ms)':>17}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for point in sweep.points:
        sc = point.scorecard
        steps = "n/a" if sc.average_steps_to_success is None else f"{sc.average_steps_to_success:.1f}"
        print(
            f"  {point.plan_horizon:>6} | "
            f"{sc.success_rate:>9.3f} | "
            f"[{point.success_ci_low:.2f}, {point.success_ci_high:.2f}]   | "
            f"{steps:>7} | "
            f"{sc.average_planning_latency_ms:>10.3f} | "
            f"[{point.latency_ci_low:.2f}, {point.latency_ci_high:.2f}]"
        )
    print()
