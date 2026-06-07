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
    the expected regime, where a hard initial state tends to hurt both at once --
    that covariance shrinks the variance of the *difference*, so the paired
    interval is tighter than the independent-proportions Agresti-Caffo interval
    of :func:`counterfactual_planning_gap`, which ignores the shared state.
    (Under negative correlation it can instead be wider; it is calibrated to the
    actual paired structure either way.) AC remains the right tool for unpaired
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
