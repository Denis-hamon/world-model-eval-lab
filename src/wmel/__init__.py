"""World Model Evaluation Lab - a lightweight, decision-oriented benchmark framework.

An independent study of evaluation methodology for action-conditioned world
models. See the disclaimer at the bottom of the Pages site for affiliation
status.
"""

from wmel.adapters.base import BenchmarkEnvironment, PlannerPolicy
from wmel.benchmark_runner import BenchmarkRunner
from wmel.experiments import (
    HorizonSweep,
    HorizonSweepPoint,
    horizon_sweep,
    print_horizon_sweep,
    to_markdown_horizon_sweep,
    wilson_interval,
)
from wmel.perturbations import (
    CompositePerturbation,
    DropNextActions,
    EnvPerturbation,
    Perturbation,
)
from wmel.report import (
    print_scorecard,
    to_json_report,
    to_markdown_report,
    to_markdown_scorecard,
)
from wmel.metrics import (
    BradleyTerryRanking,
    CorrelationResult,
    EpisodeResult,
    McNemarResult,
    Scorecard,
    ac_ci_half_width,
    action_success_rate,
    area_under_risk_coverage,
    average_planning_latency_ms,
    average_steps_to_success,
    bootstrap_correlation_ci,
    compute_scorecard,
    detectable_gap_at_n,
    holm_correction,
    kendall_tau,
    mcnemar_exact,
    newcombe_paired_diff_ci,
    paired_bradley_terry_ranking,
    perturbation_recovery_rate,
    required_n_for_half_width,
    risk_coverage_curve,
    selective_risk_at_coverage,
    spearman_rho,
)

__version__ = "0.18.0"

__all__ = [
    "BenchmarkEnvironment",
    "BenchmarkRunner",
    "BradleyTerryRanking",
    "CompositePerturbation",
    "CorrelationResult",
    "DropNextActions",
    "EnvPerturbation",
    "EpisodeResult",
    "HorizonSweep",
    "HorizonSweepPoint",
    "McNemarResult",
    "Perturbation",
    "PlannerPolicy",
    "Scorecard",
    "ac_ci_half_width",
    "action_success_rate",
    "area_under_risk_coverage",
    "average_planning_latency_ms",
    "average_steps_to_success",
    "bootstrap_correlation_ci",
    "compute_scorecard",
    "detectable_gap_at_n",
    "holm_correction",
    "horizon_sweep",
    "kendall_tau",
    "mcnemar_exact",
    "newcombe_paired_diff_ci",
    "risk_coverage_curve",
    "selective_risk_at_coverage",
    "paired_bradley_terry_ranking",
    "perturbation_recovery_rate",
    "required_n_for_half_width",
    "spearman_rho",
    "print_horizon_sweep",
    "print_scorecard",
    "to_json_report",
    "to_markdown_horizon_sweep",
    "to_markdown_report",
    "to_markdown_scorecard",
    "wilson_interval",
    "__version__",
]
