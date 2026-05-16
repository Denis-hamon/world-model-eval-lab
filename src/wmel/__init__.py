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
    EpisodeResult,
    Scorecard,
    action_success_rate,
    average_planning_latency_ms,
    average_steps_to_success,
    compute_scorecard,
    perturbation_recovery_rate,
)

__version__ = "0.10.0"

__all__ = [
    "BenchmarkEnvironment",
    "BenchmarkRunner",
    "CompositePerturbation",
    "DropNextActions",
    "EnvPerturbation",
    "EpisodeResult",
    "HorizonSweep",
    "HorizonSweepPoint",
    "Perturbation",
    "PlannerPolicy",
    "Scorecard",
    "action_success_rate",
    "average_planning_latency_ms",
    "average_steps_to_success",
    "compute_scorecard",
    "horizon_sweep",
    "perturbation_recovery_rate",
    "print_horizon_sweep",
    "print_scorecard",
    "to_json_report",
    "to_markdown_horizon_sweep",
    "to_markdown_report",
    "to_markdown_scorecard",
    "wilson_interval",
    "__version__",
]
