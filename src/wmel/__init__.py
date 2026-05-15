"""World Model Evaluation Lab - a lightweight, product-oriented benchmark framework.

This package is an independent research-to-product exploration. It is not
affiliated with AMI, Meta, the LeWorldModel project, or any of their authors.
"""

from wmel.adapters.base import BenchmarkEnvironment, PlannerPolicy
from wmel.benchmark_runner import BenchmarkRunner
from wmel.experiments import (
    HorizonSweep,
    HorizonSweepPoint,
    horizon_sweep,
    print_horizon_sweep,
    wilson_interval,
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

__version__ = "0.3.1"

__all__ = [
    "BenchmarkEnvironment",
    "BenchmarkRunner",
    "EpisodeResult",
    "HorizonSweep",
    "HorizonSweepPoint",
    "PlannerPolicy",
    "Scorecard",
    "action_success_rate",
    "average_planning_latency_ms",
    "average_steps_to_success",
    "compute_scorecard",
    "horizon_sweep",
    "perturbation_recovery_rate",
    "print_horizon_sweep",
    "wilson_interval",
    "__version__",
]
