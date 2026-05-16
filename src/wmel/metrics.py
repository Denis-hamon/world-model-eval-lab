"""Decision-grade metrics for action-conditioned world model evaluation."""

from __future__ import annotations

import math
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


@dataclass(frozen=True)
class CPGResult:
    """Counterfactual Planning Gap between two evaluation runs.

    Definition

        CPG = success_rate(oracle_dynamics) - success_rate(learned_dynamics)

    where both planners are otherwise identical (same policy, same scoring
    function, same env_factory, same episodes / horizon, same seed). The only
    thing that changes is the `dynamics=` callable passed to the planner.

    Decomposes the failure of model-based planning into two competing
    explanations:

    - **CPG > 0**: the oracle planner solves more episodes than the learned
      planner. The learned model is the bottleneck; reducing model error
      should close the gap.
    - **CPG ~ 0**: both planners succeed (or fail) at the same rate. The
      learned model is *as good as* the oracle for planning purposes; the
      remaining failures (if any) lie elsewhere (planner capacity, score
      function, env stochasticity).
    - **CPG < 0**: the learned planner outperforms the oracle. Rare, usually
      a sign that the learned model's regularisation helps the search escape
      local minima that the oracle exposes. Worth investigating, never the
      headline.

    This metric satisfies both decision-grade criteria from
    ``docs/02_metric_taxonomy.md``:

    1. Units are a fraction in [-1, 1] that translates directly to a
       deployment-time capability gap, not a model-internal score.
    2. It is computable only from closed-loop runs of both planners, not
       from the model in isolation.

    The 95% CI on the gap uses the normal approximation for the difference
    of two proportions:

        SE = sqrt( p_o(1-p_o)/n_o + p_l(1-p_l)/n_l )
        CI = gap ± 1.96 * SE

    For sample sizes typical of this framework (10-100 episodes), the
    approximation is loose near the 0/1 extremes; report Wilson-style bounds
    on each component if you need tightness there.
    """

    oracle_success_rate: float
    learned_success_rate: float
    gap: float
    n_episodes_oracle: int
    n_episodes_learned: int
    gap_ci_low: float
    gap_ci_high: float


def counterfactual_planning_gap(
    oracle_results: Sequence[EpisodeResult],
    learned_results: Sequence[EpisodeResult],
    z: float = 1.96,
) -> CPGResult:
    """Compute the Counterfactual Planning Gap between two result lists.

    Both lists must come from the same env / planner / seed / episode
    horizon; the only thing that changed between them is the dynamics
    callable passed to the planner. The function does not enforce this
    (it cannot tell from the result lists alone); the experiment script
    is responsible for setting up the comparison.

    The reported `gap` is the **raw** observed difference of success rates
    (what a reader expects to see). The reported `gap_ci_low` /
    `gap_ci_high` are the **Agresti-Caffo** 95% CI bounds on the
    difference of two binomial proportions. AC ("plus-4") adds one
    pseudo-success and one pseudo-failure to each arm before computing
    the standard normal CI, which keeps the variance positive at p=0 or
    p=1 and gives honest coverage for small n where the Wald CI's
    variance degenerates to zero. The CI is therefore **not** centred on
    `gap` for small samples - that asymmetry is the point.

    Reference: Agresti & Caffo (2000), "Simple and Effective Confidence
    Intervals for Proportions and Differences of Proportions Result from
    Adding Two Successes and Two Failures", The American Statistician.
    """
    if not oracle_results:
        raise ValueError("oracle_results must not be empty")
    if not learned_results:
        raise ValueError("learned_results must not be empty")

    n_o = len(oracle_results)
    n_l = len(learned_results)
    s_o = sum(1 for r in oracle_results if r.success)
    s_l = sum(1 for r in learned_results if r.success)
    p_o = s_o / n_o
    p_l = s_l / n_l
    gap = p_o - p_l

    # Agresti-Caffo plus-4 adjustment: one pseudo-success and one
    # pseudo-failure per arm, so the variance never collapses at p in {0, 1}.
    tilde_p_o = (s_o + 1.0) / (n_o + 2.0)
    tilde_p_l = (s_l + 1.0) / (n_l + 2.0)
    tilde_gap = tilde_p_o - tilde_p_l
    se = math.sqrt(
        tilde_p_o * (1.0 - tilde_p_o) / (n_o + 2.0)
        + tilde_p_l * (1.0 - tilde_p_l) / (n_l + 2.0)
    )
    half_width = z * se
    return CPGResult(
        oracle_success_rate=p_o,
        learned_success_rate=p_l,
        gap=gap,
        n_episodes_oracle=n_o,
        n_episodes_learned=n_l,
        gap_ci_low=tilde_gap - half_width,
        gap_ci_high=tilde_gap + half_width,
    )


CPG_VERDICT_MODEL_BOTTLENECK = "MODEL BOTTLENECK"
CPG_VERDICT_LEARNED_OUTPERFORMS = "LEARNED OUTPERFORMS ORACLE"
CPG_VERDICT_PLANNER_BOTTLENECK = "PLANNER BOTTLENECK"
CPG_VERDICT_MODEL_AS_GOOD_AS_ORACLE = "MODEL AS GOOD AS ORACLE"
CPG_VERDICT_INCONCLUSIVE = "INCONCLUSIVE"


def cpg_verdict(cpg: CPGResult, both_extreme_tol: float = 0.05) -> str:
    """Decision rule for CPG that respects the confidence interval.

    A CPG reported without a significance gate is misleading: a `gap=+0.1`
    from n=10 is indistinguishable from noise but would otherwise read as
    "MODEL BOTTLENECK". This function gates the verdict on whether the
    Agresti-Caffo CI on the gap actually crosses zero.

    Branches:

    - **MODEL BOTTLENECK**: `gap_ci_low > 0`. Oracle reliably better than
      learned; improving the model should close the gap.
    - **LEARNED OUTPERFORMS ORACLE**: `gap_ci_high < 0`. Rare; usually a
      regularisation artifact. Worth investigating, never the headline.
    - **PLANNER BOTTLENECK**: CI crosses 0 AND both success rates are
      within `both_extreme_tol` of 0. Neither planner solves the task; the
      framework needs a stronger search procedure (CEM, iLQR, learned
      planner), not a stronger model.
    - **MODEL AS GOOD AS ORACLE**: CI crosses 0 AND both success rates are
      within `both_extreme_tol` of 1. The learned model is as good as the
      oracle for planning purposes on this task.
    - **INCONCLUSIVE**: CI crosses 0 in a middle-of-the-road regime. The
      sample size is insufficient to discriminate. Report and run more
      episodes.
    """
    if cpg.gap_ci_low > 0:
        return CPG_VERDICT_MODEL_BOTTLENECK
    if cpg.gap_ci_high < 0:
        return CPG_VERDICT_LEARNED_OUTPERFORMS
    if (
        cpg.oracle_success_rate <= both_extreme_tol
        and cpg.learned_success_rate <= both_extreme_tol
    ):
        return CPG_VERDICT_PLANNER_BOTTLENECK
    if (
        cpg.oracle_success_rate >= 1.0 - both_extreme_tol
        and cpg.learned_success_rate >= 1.0 - both_extreme_tol
    ):
        return CPG_VERDICT_MODEL_AS_GOOD_AS_ORACLE
    return CPG_VERDICT_INCONCLUSIVE
