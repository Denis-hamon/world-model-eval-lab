"""Decision-grade metrics for action-conditioned world model evaluation."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from statistics import fmean
from typing import Callable, Hashable, Iterable, Mapping, Sequence


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
    confidence: float | None = field(default=None, kw_only=True)
    """Optional deployable confidence the policy attached to this episode's
    decision (higher = more trusted), used only by the selective-prediction
    metrics. ``None`` means the policy exposed no abstention signal."""

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


def paired_bootstrap_gap_ci(
    oracle_results: Sequence[EpisodeResult],
    learned_results: Sequence[EpisodeResult],
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI on the CPG when the two arms are *paired*.

    Use this when episode ``k`` of each list starts from the **same** initial
    state -- which is the case for varied-initial-state runs, where both arms
    share the per-episode task seed (``experiments/_seeding.py``). The two
    lists must then be equal length and index-aligned.

    Resampling episode *indices* (the same index drawn for both arms), rather
    than resampling the two arms independently, preserves the within-episode
    correlation between the arms. When the arms are **positively** correlated --
    a hard initial state tending to hurt both at once -- that covariance shrinks
    the variance of the *difference* and the paired interval is tighter than the
    independent-proportions Agresti-Caffo interval of
    :func:`counterfactual_planning_gap`, which ignores the shared state; under
    negative or zero correlation it is instead wider or comparable. It is
    calibrated to whatever the actual paired structure is (which is an empirical
    matter: on the cells evaluated here the within-pair correlation is near zero,
    so the two intervals nearly coincide -- see ``experiments/paired_intervals_audit.py``). AC remains the right tool for unpaired
    or boundary (``0/n``) cells; this is the right tool for the non-degenerate
    paired cells where the two arms can both succeed or both fail on the same
    start.

    Returns ``(gap_point, ci_low, ci_high)``: the raw paired difference of
    success rates and the ``[alpha/2, 1 - alpha/2]`` percentile bootstrap
    bounds. Deterministic given ``seed``.
    """
    if len(oracle_results) != len(learned_results):
        raise ValueError(
            "paired bootstrap requires equal-length, index-aligned arms "
            f"(got {len(oracle_results)} oracle, {len(learned_results)} learned)"
        )
    if not oracle_results:
        raise ValueError("results must not be empty")
    if n_boot < 1:
        raise ValueError("n_boot must be >= 1")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1)")

    o = [1.0 if r.success else 0.0 for r in oracle_results]
    l = [1.0 if r.success else 0.0 for r in learned_results]
    n = len(o)
    gap_point = fmean(o) - fmean(l)

    rng = random.Random(seed)
    gaps: list[float] = []
    for _ in range(n_boot):
        diff_sum = 0.0
        for _ in range(n):
            j = rng.randrange(n)
            diff_sum += o[j] - l[j]
        gaps.append(diff_sum / n)
    gaps.sort()

    lo_idx = max(0, min(int((alpha / 2.0) * n_boot), n_boot - 1))
    hi_idx = max(0, min(int((1.0 - alpha / 2.0) * n_boot) - 1, n_boot - 1))
    return gap_point, gaps[lo_idx], gaps[hi_idx]


def _paired_success_arrays(
    oracle_results: Sequence[EpisodeResult],
    learned_results: Sequence[EpisodeResult],
) -> tuple[list[int], list[int]]:
    """Validate the paired contract and return 0/1 success arrays per arm."""
    if len(oracle_results) != len(learned_results):
        raise ValueError(
            "paired methods require equal-length, index-aligned arms "
            f"(got {len(oracle_results)} oracle, {len(learned_results)} learned)"
        )
    if not oracle_results:
        raise ValueError("results must not be empty")
    o = [1 if r.success else 0 for r in oracle_results]
    l = [1 if r.success else 0 for r in learned_results]
    return o, l


def _wilson_bounds(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a single proportion (stdlib).

    Unlike Wald, it does not collapse at ``successes`` in ``{0, total}``, which
    is exactly why it is the right building block for Newcombe's paired
    difference interval at the boundary proportions this framework hits.
    """
    if total == 0:
        return 0.0, 1.0
    p = successes / total
    z2 = z * z
    denom = 1.0 + z2 / total
    centre = (p + z2 / (2.0 * total)) / denom
    half = (z * math.sqrt(p * (1.0 - p) / total + z2 / (4.0 * total * total))) / denom
    return centre - half, centre + half


@dataclass(frozen=True)
class McNemarResult:
    """Exact McNemar test for two *paired* binary arms (oracle vs learned).

    The two arms are evaluated on the same per-episode initial state, so each
    episode is a matched pair. McNemar conditions on the *discordant* pairs --
    episodes where exactly one arm succeeds -- and asks whether their split
    departs from 50/50. Unlike the Agresti-Caffo interval (which treats the two
    arms as *independent* proportions and so ignores the pairing), this is the
    textbook test for the paired binary design; being exact, it stays valid at
    the boundary (a learned arm at ``0/n`` yields only oracle-only discordant
    pairs and reduces to a one-sided binomial / sign test).

    The counts form the 2x2 paired table:

        both         : episodes both arms solve
        oracle_only  : oracle solves, learned fails   (the ``b`` cell)
        learned_only : learned solves, oracle fails   (the ``c`` cell)
        neither      : episodes neither arm solves

    ``p_value`` is the two-sided exact probability under H0 (equal marginal
    success), i.e. ``b ~ Binomial(b + c, 1/2)``; it is ``1.0`` when there are no
    discordant pairs (the data carry no paired evidence either way).
    """

    both: int
    oracle_only: int
    learned_only: int
    neither: int
    n_discordant: int
    p_value: float


def mcnemar_exact(
    oracle_results: Sequence[EpisodeResult],
    learned_results: Sequence[EpisodeResult],
) -> McNemarResult:
    """Exact two-sided McNemar test on the paired (oracle, learned) arms.

    See :class:`McNemarResult`. The two lists must be equal length and
    index-aligned (episode ``k`` is the same initial state in both arms) -- the
    same paired contract as :func:`paired_bootstrap_gap_ci`.
    """
    o, l = _paired_success_arrays(oracle_results, learned_results)
    both = sum(1 for a, b in zip(o, l) if a == 1 and b == 1)
    oracle_only = sum(1 for a, b in zip(o, l) if a == 1 and b == 0)
    learned_only = sum(1 for a, b in zip(o, l) if a == 0 and b == 1)
    neither = sum(1 for a, b in zip(o, l) if a == 0 and b == 0)
    n_disc = oracle_only + learned_only
    if n_disc == 0:
        p_value = 1.0
    else:
        k = min(oracle_only, learned_only)
        # Two-sided exact binomial test at p=1/2: double the smaller tail,
        # capped at 1.0 (the doubling can exceed 1 when oracle_only == learned_only).
        tail = sum(math.comb(n_disc, i) for i in range(k + 1)) / (2.0 ** n_disc)
        p_value = min(1.0, 2.0 * tail)
    return McNemarResult(
        both=both,
        oracle_only=oracle_only,
        learned_only=learned_only,
        neither=neither,
        n_discordant=n_disc,
        p_value=p_value,
    )


def newcombe_paired_diff_ci(
    oracle_results: Sequence[EpisodeResult],
    learned_results: Sequence[EpisodeResult],
    z: float = 1.96,
) -> tuple[float, float, float]:
    """Newcombe (1998) CI for the difference of two *paired* proportions.

    Returns ``(diff_point, ci_low, ci_high)`` for ``p_oracle - p_learned`` via
    Newcombe's square-and-add method: combine each arm's Wilson score interval
    with a correction for the paired correlation ``phi`` estimated from the 2x2
    table. The interval is bounded in ``[-1, 1]`` and stays well-defined at the
    boundary (the Wilson components never collapse), so it is the closed-form
    paired analogue of the Agresti-Caffo interval and the companion to the
    nonparametric :func:`paired_bootstrap_gap_ci`. Signature mirrors that
    function; the two lists must be equal length and index-aligned.

    Reference: Newcombe (1998), "Improved confidence intervals for the difference
    between binomial proportions based on paired data", Statistics in Medicine
    17(22):2635-2650 (the phi-corrected square-and-add on Wilson intervals).
    """
    o, l = _paired_success_arrays(oracle_results, learned_results)
    n = len(o)
    a = sum(1 for x, y in zip(o, l) if x == 1 and y == 1)  # both
    b = sum(1 for x, y in zip(o, l) if x == 1 and y == 0)  # oracle only
    c = sum(1 for x, y in zip(o, l) if x == 0 and y == 1)  # learned only
    d = sum(1 for x, y in zip(o, l) if x == 0 and y == 0)  # neither
    p1 = (a + b) / n  # oracle success rate
    p2 = (a + c) / n  # learned success rate
    diff = p1 - p2

    l1, u1 = _wilson_bounds(a + b, n, z)
    l2, u2 = _wilson_bounds(a + c, n, z)

    # Paired correlation correction; 0 when any margin is degenerate.
    margin = (a + b) * (c + d) * (a + c) * (b + d)
    phi = ((a * d - b * c) / math.sqrt(margin)) if margin > 0 else 0.0

    lower = diff - math.sqrt(
        max(0.0, (p1 - l1) ** 2 - 2.0 * phi * (p1 - l1) * (u2 - p2) + (u2 - p2) ** 2)
    )
    upper = diff + math.sqrt(
        max(0.0, (u1 - p1) ** 2 - 2.0 * phi * (u1 - p1) * (p2 - l2) + (p2 - l2) ** 2)
    )
    return diff, lower, upper


def holm_correction(pvalues: Sequence[float]) -> list[float]:
    """Holm-Bonferroni step-down adjusted p-values (family-wise error control).

    Given a family of raw p-values -- e.g. one :func:`mcnemar_exact` test per
    evaluated cell -- returns adjusted p-values in the **same order** as the
    input. Controls the family-wise error rate without assuming independence and
    is uniformly more powerful than plain Bonferroni. Compare each adjusted value
    to ``alpha`` as usual. This is the multiplicity correction a heterogeneous
    grid of (env, planner, model) verdicts needs before any cell is called
    significant.
    """
    m = len(pvalues)
    if m == 0:
        return []
    for p in pvalues:
        if not (0.0 <= p <= 1.0):
            raise ValueError(f"p-values must be in [0, 1] (got {p})")
    order = sorted(range(m), key=lambda i: pvalues[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        # (m - rank) multiplier for the rank-th smallest (0-indexed); enforce
        # monotonic non-decreasing adjusted values down the sorted order.
        running = max(running, min(1.0, (m - rank) * pvalues[idx]))
        adjusted[idx] = running
    return adjusted


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


def ac_ci_half_width(
    oracle_success_rate: float,
    learned_success_rate: float,
    n_per_arm: int,
    z: float = 1.96,
) -> float:
    """Agresti-Caffo plus-4 half-width of the CPG CI at a given per-arm n.

    This is the same standard error the CPG CI uses (see
    ``counterfactual_planning_gap``), evaluated at hypothetical success rates
    and an equal per-arm episode count. It answers "how tight would the
    interval be if I ran ``n_per_arm`` episodes per arm and observed these
    rates?" without running anything.

    The plus-4 adjustment uses the rates to form pseudo-counts, so the inputs
    are the *expected* proportions, not integer successes. Equal n per arm is
    assumed because that is the design a practitioner controls.
    """
    if n_per_arm < 1:
        raise ValueError("n_per_arm must be >= 1")
    tilde_p_o = (oracle_success_rate * n_per_arm + 1.0) / (n_per_arm + 2.0)
    tilde_p_l = (learned_success_rate * n_per_arm + 1.0) / (n_per_arm + 2.0)
    se = math.sqrt(
        tilde_p_o * (1.0 - tilde_p_o) / (n_per_arm + 2.0)
        + tilde_p_l * (1.0 - tilde_p_l) / (n_per_arm + 2.0)
    )
    return z * se


def required_n_for_half_width(
    oracle_success_rate: float,
    learned_success_rate: float,
    target_half_width: float,
    z: float = 1.96,
    n_max: int = 100_000,
) -> int | None:
    """Smallest equal per-arm n whose AC CI half-width is <= target.

    Power-analysis companion to the CPG verdict: given a hypothesised pair of
    success rates and a target precision (e.g. a half-width of 0.05 so the
    interval is +/-5 percentage points), return the per-arm episode count
    needed. Monotone in n, so a simple search suffices. Returns None if even
    ``n_max`` is insufficient (only possible for absurdly small targets).

    This is the quantity that turns CPG from a verdict into a planning tool:
    before running a comparison, a practitioner reads off how many episodes
    are needed to make the verdict gate able to fire at the precision they
    care about. A point-estimate leaderboard cannot answer this.
    """
    if target_half_width <= 0:
        raise ValueError("target_half_width must be > 0")
    lo, hi = 1, n_max
    if ac_ci_half_width(oracle_success_rate, learned_success_rate, hi, z) > target_half_width:
        return None
    # Binary search for the smallest n meeting the target (half-width is
    # monotone decreasing in n for fixed rates).
    while lo < hi:
        mid = (lo + hi) // 2
        if ac_ci_half_width(oracle_success_rate, learned_success_rate, mid, z) <= target_half_width:
            hi = mid
        else:
            lo = mid + 1
    return lo


def detectable_gap_at_n(
    oracle_success_rate: float,
    learned_success_rate: float,
    n_per_arm: int,
    z: float = 1.96,
) -> bool:
    """Whether the CPG verdict gate would clear zero at this n and these rates.

    Returns True iff the AC CI on the gap does not straddle zero, i.e. the
    five-branch verdict would commit (MODEL BOTTLENECK or LEARNED OUTPERFORMS)
    rather than return INCONCLUSIVE, under the hypothesised rates at this
    sample size. The companion check to ``required_n_for_half_width`` when the
    question is "is this n enough to *decide*?" rather than "to reach this
    precision?".
    """
    tilde_p_o = (oracle_success_rate * n_per_arm + 1.0) / (n_per_arm + 2.0)
    tilde_p_l = (learned_success_rate * n_per_arm + 1.0) / (n_per_arm + 2.0)
    tilde_gap = tilde_p_o - tilde_p_l
    hw = ac_ci_half_width(oracle_success_rate, learned_success_rate, n_per_arm, z)
    return (tilde_gap - hw) > 0 or (tilde_gap + hw) < 0


def selective_risk_at_coverage(
    results: Sequence[EpisodeResult],
    coverage: float,
) -> float | None:
    """Selective risk at a target coverage, for a policy with a reject option.

    Rank episodes by ``confidence`` (descending), keep the most-confident
    ``coverage`` fraction (the rest are deferred / abstained), and return the
    error rate (``1 - success_rate``) on the kept set. This is the decision-grade
    way to score a world model used as a *verifier*: it should act where it is
    confident and defer elsewhere, so the question is "how low is the risk on the
    slice it chooses to act on?" rather than "what is its average fidelity?".

    Decision-grade: the unit is a risk fraction in ``[0, 1]`` at a chosen
    coverage fraction, computable only from a closed-loop run whose policy
    exposed a per-episode ``confidence``.

    Returns ``None`` if no episode carries a ``confidence`` (nothing to rank on).
    Raises ``ValueError`` if ``coverage`` is not in ``(0, 1]``.
    """
    if not 0.0 < coverage <= 1.0:
        raise ValueError("coverage must be in (0, 1]")
    scored = [r for r in results if r.confidence is not None]
    if not scored:
        return None
    ranked = sorted(scored, key=lambda r: r.confidence, reverse=True)
    k = max(1, round(coverage * len(ranked)))
    kept = ranked[:k]
    return 1.0 - sum(1 for r in kept if r.success) / len(kept)


def risk_coverage_curve(
    results: Sequence[EpisodeResult],
    coverages: Sequence[float] = (1.0, 0.9, 0.75, 0.5, 0.25, 0.1),
) -> list[tuple[float, float]] | None:
    """The risk-coverage curve: selective risk at several coverages.

    A monotone-decreasing curve (risk falls as coverage shrinks) means the
    confidence signal is useful for abstention; a flat or non-monotone curve
    means it is not. Returns ``None`` if no episode carries a ``confidence``.
    """
    scored = [r for r in results if r.confidence is not None]
    if not scored:
        return None
    return [(c, selective_risk_at_coverage(scored, c)) for c in coverages]


def area_under_risk_coverage(results: Sequence[EpisodeResult]) -> float | None:
    """Area under the risk-coverage curve (AURC); lower is better.

    The standard scalar summary of a selective predictor: the mean selective
    risk as coverage sweeps from the single most-confident episode up to full
    coverage. A confidence that orders episodes perfectly by correctness attains
    the lowest AURC the data allow; an uninformative confidence sits at the base
    error rate. Returns ``None`` if no episode carries a ``confidence``.
    """
    scored = [r for r in results if r.confidence is not None]
    if not scored:
        return None
    ranked = sorted(scored, key=lambda r: r.confidence, reverse=True)
    errors = 0
    risks: list[float] = []
    for i, r in enumerate(ranked, start=1):
        if not r.success:
            errors += 1
        risks.append(errors / i)
    return sum(risks) / len(risks)


@dataclass(frozen=True)
class BradleyTerryRanking:
    """A Bradley-Terry ranking of several models from *paired* per-episode runs.

    CPG (:func:`counterfactual_planning_gap`) compares one learned model to the
    oracle. When the question is instead "which of these N models plans best,
    and how confident is that ordering?", the right object is a ranking with
    uncertainty, not N separate gaps. This dataclass is that ranking.

    Fields
    ------
    models
        Model names in input order.
    strengths
        Bradley-Terry strengths, normalised to sum to 1 (higher is better). For
        two models with ``prior=0`` the strength is *exactly* the empirical win
        fraction; the default ``prior=0.5`` shrinks it slightly toward ``0.5``.
    ranks
        Competition rank, ``1`` = best. Tied strengths share the lower rank.
    strength_ci
        ``model -> (low, high)`` percentile bootstrap CI on the strength.
    rank_ci
        ``model -> (best, worst)`` plausible rank from the bootstrap: the
        ``alpha/2`` and ``1 - alpha/2`` percentiles of the model's rank across
        resamples. A model whose ``rank_ci`` overlaps another's is *not*
        distinguishable from it at this sample size -- the leaderboard analogue
        of an ``INCONCLUSIVE`` CPG verdict.
    win_matrix
        ``i -> j -> k``: the number of episodes on which model ``i`` strictly
        beat model ``j`` (raw counts, excluding the prior).
    n_decisive
        ``i -> j -> k``: decisive comparisons between ``i`` and ``j`` (wins
        either way; ties excluded). Symmetric.
    n_episodes, n_models, n_boot, prior
        The run configuration, retained so a reader can audit the ranking.
    """

    models: tuple[str, ...]
    strengths: dict[str, float]
    ranks: dict[str, int]
    strength_ci: dict[str, tuple[float, float]]
    rank_ci: dict[str, tuple[int, int]]
    win_matrix: dict[str, dict[str, int]]
    n_decisive: dict[str, dict[str, int]]
    n_episodes: int
    n_models: int
    n_boot: int
    prior: float


def _default_outcome(r: EpisodeResult) -> float:
    return 1.0 if r.success else 0.0


def _bt_wins_from_indices(
    m: int,
    outcomes: list[list[float]],
    indices: Sequence[int],
    higher_is_better: bool,
) -> list[list[int]]:
    """Pairwise win counts over the episodes in ``indices`` (positions, paired).

    ``outcomes[a][k]`` is model ``a``'s scalar outcome on episode ``k``. On each
    episode the comparison is between models evaluated on the *same* initial
    state (the design is paired), so every pairwise comparison happens under
    identical conditions -- the assumption Bradley-Terry needs. Ties (equal
    outcome, e.g. both arms fail) carry no ordering information and are skipped.
    """
    wins = [[0] * m for _ in range(m)]
    for k in indices:
        for a in range(m):
            oa = outcomes[a][k]
            for b in range(a + 1, m):
                ob = outcomes[b][k]
                if oa == ob:
                    continue
                a_beats_b = (oa > ob) if higher_is_better else (oa < ob)
                if a_beats_b:
                    wins[a][b] += 1
                else:
                    wins[b][a] += 1
    return wins


def _fit_bradley_terry(
    m: int,
    wins: list[list[int]],
    prior: float,
    max_iter: int,
    tol: float,
) -> list[float]:
    """Bradley-Terry strengths via the MM (Zermelo) iteration, normalised to 1.

    ``prior`` adds ``prior`` pseudo-wins to *each* ordered pair before fitting.
    This is the same philosophy as the Agresti-Caffo plus-4 adjustment used
    everywhere else in this module: a small pseudo-count keeps the estimator
    finite at the boundary. Here the boundary is *separation* -- a model that
    wins (or loses) all of its decisive comparisons, for which the unregularised
    Bradley-Terry MLE diverges. ``prior=0`` recovers the plain MLE, which may not
    converge under separation; the default ``prior=0.5`` makes the comparison
    graph complete so the iteration always converges to a finite ranking. This
    matters in the bootstrap, where a resample frequently isolates a model.
    """
    pi = [1.0 / m] * m
    w_reg = [
        sum(wins[i][j] for j in range(m) if j != i) + prior * (m - 1) for i in range(m)
    ]
    n_reg = [
        [
            (wins[i][j] + wins[j][i] + 2.0 * prior) if j != i else 0.0
            for j in range(m)
        ]
        for i in range(m)
    ]
    for _ in range(max_iter):
        new = [0.0] * m
        for i in range(m):
            denom = 0.0
            for j in range(m):
                if j == i:
                    continue
                pair = pi[i] + pi[j]
                if pair > 0.0:
                    denom += n_reg[i][j] / pair
            new[i] = w_reg[i] / denom if denom > 0 else pi[i]
        total = sum(new)
        if total > 0:
            new = [x / total for x in new]
        delta = max(abs(new[i] - pi[i]) for i in range(m))
        pi = new
        if delta < tol:
            break
    return pi


def _bt_ranks(pi: Sequence[float], tol: float = 1e-9) -> list[int]:
    """Competition ranks from strengths (1 = best, ties share the lower rank)."""
    return [1 + sum(1 for x in pi if x > pi[i] + tol) for i in range(len(pi))]


def _percentile_bounds(sorted_values: Sequence[float], alpha: float, n_boot: int):
    lo_idx = max(0, min(int((alpha / 2.0) * n_boot), n_boot - 1))
    hi_idx = max(0, min(int((1.0 - alpha / 2.0) * n_boot) - 1, n_boot - 1))
    return sorted_values[lo_idx], sorted_values[hi_idx]


def paired_bradley_terry_ranking(
    results: Mapping[str, Sequence[EpisodeResult]],
    *,
    outcome: Callable[[EpisodeResult], float] = _default_outcome,
    higher_is_better: bool = True,
    groups: Sequence[Hashable] | None = None,
    prior: float = 0.5,
    n_boot: int = 2_000,
    alpha: float = 0.05,
    seed: int = 0,
    max_iter: int = 1_000,
    tol: float = 1e-10,
) -> BradleyTerryRanking:
    """Rank N models by a Bradley-Terry fit to their *paired* per-episode runs.

    This generalises the two-arm, paired comparison of
    :func:`paired_bootstrap_gap_ci` to a full N-model leaderboard with
    uncertainty on the *ranking* itself, in the spirit of the pairwise,
    matched-condition evaluation used for generalist-policy arenas (Atreya et
    al., "RoboArena", 2025). It is the leaderboard companion to the CPG
    power-analysis helpers: where ``detectable_gap_at_n`` asks whether one gap
    clears zero, this asks whether a *ranking* is distinguishable from noise.

    Design contract. ``results`` maps each model name to its per-episode results.
    All lists must be the same length and **index-aligned**: episode ``k`` must
    start from the same initial state for every model (e.g. the varied-init
    arms paired by per-episode seed, ``experiments/_seeding.py``). Because every
    pairwise comparison is then made on a shared initial state, each comparison
    satisfies the identical-conditions assumption Bradley-Terry requires -- the
    very assumption RoboArena flags as violated when evaluators pick disparate
    tasks. ``outcome`` maps an ``EpisodeResult`` to the scalar that is compared
    per episode (default: ``1.0`` on success else ``0.0``); set
    ``higher_is_better=False`` for a cost-like outcome. Equal outcomes are ties
    and contribute no ordering information.

    Uncertainty. The point ranking uses every episode once. The CIs come from a
    paired percentile bootstrap that resamples episode *indices* (the same draw
    applied to all models, preserving pairing) and refits the ranking each time.
    When ``groups`` is given (one task label per episode), the bootstrap
    resamples *within* each group, so the rank interval reflects the task mix
    rather than a resample that happens to over-weight one task. This is lighter
    than RoboArena's task-aware ranking: it recalibrates the rank *interval*
    only, it does not change the point ranking (already condition-matched by the
    paired design, since every model is scored on the same per-episode state).

    Returns a :class:`BradleyTerryRanking`. Deterministic given ``seed``.

    Reference: Hunter (2004), "MM algorithms for generalized Bradley-Terry
    models", Annals of Statistics, for the iteration used here.
    """
    models = tuple(results.keys())
    m = len(models)
    if m < 2:
        raise ValueError("need at least two models to rank")
    lengths = {len(results[name]) for name in models}
    if len(lengths) != 1:
        raise ValueError(
            "paired ranking requires equal-length, index-aligned result lists "
            f"(got lengths {sorted(lengths)})"
        )
    n = lengths.pop()
    if n == 0:
        raise ValueError("result lists must not be empty")
    if groups is not None and len(groups) != n:
        raise ValueError(
            f"groups must have one label per episode (got {len(groups)} for n={n})"
        )
    if prior < 0:
        raise ValueError("prior must be >= 0")
    if n_boot < 1:
        raise ValueError("n_boot must be >= 1")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1)")

    outcomes = [[float(outcome(r)) for r in results[name]] for name in models]

    # Point ranking over every episode once.
    point_wins = _bt_wins_from_indices(m, outcomes, range(n), higher_is_better)
    if prior == 0.0:
        # The unregularised Bradley-Terry MLE exists only when the comparison
        # graph is connected: every model must have at least one decisive win
        # and one decisive loss (Ford, 1957). Otherwise the estimate is not
        # identifiable and would be silently arbitrary. Refuse rather than
        # mislead; the default prior > 0 makes the ranking well-defined.
        for i in range(m):
            wins_i = sum(point_wins[i][j] for j in range(m) if j != i)
            losses_i = sum(point_wins[j][i] for j in range(m) if j != i)
            if wins_i == 0 or losses_i == 0:
                raise ValueError(
                    "with prior=0 the Bradley-Terry MLE is undefined unless every "
                    "model has at least one decisive win and one decisive loss "
                    f"(model {models[i]!r} has {wins_i} wins, {losses_i} losses); "
                    "use the default prior > 0 for a regularised ranking"
                )
    point_pi = _fit_bradley_terry(m, point_wins, prior, max_iter, tol)
    point_ranks = _bt_ranks(point_pi)

    # Index groups for the (optionally stratified) paired bootstrap.
    if groups is None:
        group_indices = [list(range(n))]
    else:
        buckets: dict[Hashable, list[int]] = {}
        for k, g in enumerate(groups):
            buckets.setdefault(g, []).append(k)
        group_indices = list(buckets.values())

    rng = random.Random(seed)
    strength_samples: list[list[float]] = [[] for _ in range(m)]
    rank_samples: list[list[int]] = [[] for _ in range(m)]
    for _ in range(n_boot):
        idx: list[int] = []
        for bucket in group_indices:
            size = len(bucket)
            for _ in range(size):
                idx.append(bucket[rng.randrange(size)])
        wins_b = _bt_wins_from_indices(m, outcomes, idx, higher_is_better)
        pi_b = _fit_bradley_terry(m, wins_b, prior, max_iter, tol)
        ranks_b = _bt_ranks(pi_b)
        for i in range(m):
            strength_samples[i].append(pi_b[i])
            rank_samples[i].append(ranks_b[i])

    strengths = {models[i]: point_pi[i] for i in range(m)}
    ranks = {models[i]: point_ranks[i] for i in range(m)}
    strength_ci: dict[str, tuple[float, float]] = {}
    rank_ci: dict[str, tuple[int, int]] = {}
    for i in range(m):
        s_sorted = sorted(strength_samples[i])
        lo, hi = _percentile_bounds(s_sorted, alpha, n_boot)
        strength_ci[models[i]] = (lo, hi)
        r_sorted = sorted(rank_samples[i])
        r_lo, r_hi = _percentile_bounds(r_sorted, alpha, n_boot)
        rank_ci[models[i]] = (int(r_lo), int(r_hi))

    win_matrix = {
        models[i]: {models[j]: point_wins[i][j] for j in range(m) if j != i}
        for i in range(m)
    }
    n_decisive = {
        models[i]: {
            models[j]: point_wins[i][j] + point_wins[j][i] for j in range(m) if j != i
        }
        for i in range(m)
    }

    return BradleyTerryRanking(
        models=models,
        strengths=strengths,
        ranks=ranks,
        strength_ci=strength_ci,
        rank_ci=rank_ci,
        win_matrix=win_matrix,
        n_decisive=n_decisive,
        n_episodes=n,
        n_models=m,
        n_boot=n_boot,
        prior=prior,
    )


# --- Rank correlation (for offline-metric vs downstream-performance studies) --

@dataclass(frozen=True)
class CorrelationResult:
    """A rank correlation with a bootstrap confidence interval.

    Built for the question "does a cheap offline metric predict downstream
    decision quality?": correlate one value per (model, env, planner) cell
    against its CPG / success and report the strength with an honest interval.
    Rank-based because at the handful-of-cells sample sizes this is used for, a
    monotone (not linear) relationship on incomparable scales is what matters.
    ``n_boot`` is the number of *valid* (non-degenerate) resamples actually used,
    which can be fewer than the count requested.
    """

    rho: float
    ci_low: float
    ci_high: float
    n_pairs: int
    method: str
    n_boot: int


def _rankdata(xs: Sequence[float]) -> list[float]:
    """Average (fractional) ranks with tie handling; ranks are 1-based."""
    n = len(xs)
    order = sorted(range(n), key=lambda i: xs[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0  # mean of the 1-based positions i..j
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _pearson(a: Sequence[float], b: Sequence[float]) -> float:
    """Pearson correlation; raises ValueError if either side has zero variance."""
    mean_a, mean_b = fmean(a), fmean(b)
    num = sum((ai - mean_a) * (bi - mean_b) for ai, bi in zip(a, b))
    den = math.sqrt(sum((ai - mean_a) ** 2 for ai in a)) * math.sqrt(
        sum((bi - mean_b) ** 2 for bi in b)
    )
    if den == 0.0:
        raise ValueError("correlation undefined: zero variance in an input")
    return num / den


def spearman_rho(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Spearman rank correlation in [-1, 1] (Pearson on average ranks, tie-safe).

    Raises ValueError on mismatched lengths, fewer than two points, or a
    constant input (correlation undefined).
    """
    if len(xs) != len(ys):
        raise ValueError(f"length mismatch: {len(xs)} vs {len(ys)}")
    if len(xs) < 2:
        raise ValueError("need at least two pairs")
    return _pearson(_rankdata(xs), _rankdata(ys))


def kendall_tau(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Kendall tau-b in [-1, 1] (tie-corrected). O(n^2); robust at very small n.

    Raises ValueError on mismatched lengths, fewer than two points, or a
    degenerate denominator (a constant input).
    """
    if len(xs) != len(ys):
        raise ValueError(f"length mismatch: {len(xs)} vs {len(ys)}")
    n = len(xs)
    if n < 2:
        raise ValueError("need at least two pairs")
    n0 = n * (n - 1) // 2
    nc = nd = n1 = n2 = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            if dx == 0 and dy == 0:
                n1 += 1
                n2 += 1
            elif dx == 0:
                n1 += 1
            elif dy == 0:
                n2 += 1
            elif (dx > 0) == (dy > 0):
                nc += 1
            else:
                nd += 1
    den = math.sqrt((n0 - n1) * (n0 - n2))
    if den == 0.0:
        raise ValueError("Kendall tau-b undefined: a constant input")
    return (nc - nd) / den


def bootstrap_correlation_ci(
    xs: Sequence[float],
    ys: Sequence[float],
    *,
    method: str = "spearman",
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> CorrelationResult:
    """Rank correlation with a paired percentile bootstrap CI.

    Resamples cell indices with replacement (the pair ``(x_i, y_i)`` kept
    together, like :func:`paired_bootstrap_gap_ci`) and recomputes the
    correlation each draw. Degenerate resamples (a constant input, which makes
    the correlation undefined) are skipped; the reported ``n_boot`` is the number
    of valid resamples. The interval is therefore conditional on non-degenerate
    resamples -- for a near-constant arm at very small n this can make it
    optimistically narrow. Deterministic given ``seed``.
    """
    if len(xs) != len(ys):
        raise ValueError(f"length mismatch: {len(xs)} vs {len(ys)}")
    n = len(xs)
    if n < 2:
        raise ValueError("need at least two pairs")
    if n_boot < 1:
        raise ValueError("n_boot must be >= 1")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1)")
    fns = {"spearman": spearman_rho, "kendall": kendall_tau}
    if method not in fns:
        raise ValueError(f"method must be one of {sorted(fns)} (got {method!r})")
    fn = fns[method]

    point = fn(xs, ys)
    rng = random.Random(seed)
    rhos: list[float] = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        try:
            rhos.append(fn([xs[i] for i in idx], [ys[i] for i in idx]))
        except ValueError:
            continue  # degenerate resample (a constant arm): no information
    if len(rhos) < 2:
        raise ValueError("correlation bootstrap degenerate: too few valid resamples")
    rhos.sort()
    m = len(rhos)
    lo = rhos[max(0, min(int((alpha / 2.0) * m), m - 1))]
    hi = rhos[max(0, min(int((1.0 - alpha / 2.0) * m) - 1, m - 1))]
    return CorrelationResult(
        rho=point, ci_low=lo, ci_high=hi, n_pairs=n, method=method, n_boot=m
    )
