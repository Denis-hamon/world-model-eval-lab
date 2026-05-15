"""Reporting helpers: print a scorecard, dump a JSON report."""

from __future__ import annotations

from dataclasses import asdict
from typing import Sequence

from wmel.metrics import EpisodeResult, Scorecard


def _fmt(value: float | None, suffix: str = "") -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}{suffix}"


def print_scorecard(scorecard: Scorecard) -> None:
    """Print a human-readable scorecard to stdout."""
    print(f"Scorecard: {scorecard.policy_name}")
    print("-" * 48)
    print(f"  episodes                       : {scorecard.episodes}")
    print(f"  action success rate            : {_fmt(scorecard.success_rate)}")
    print(f"  average steps to success       : {_fmt(scorecard.average_steps_to_success)}")
    print(f"  planning latency per call (ms) : {_fmt(scorecard.average_planning_latency_ms)}")
    print(f"  perturbation recovery rate     : {_fmt(scorecard.perturbation_recovery_rate)}")
    print(f"  average compute per decision   : {_fmt(scorecard.average_compute_per_decision)}")
    for name, value in scorecard.extras.items():
        print(f"  {name:<31}: {_fmt(value)}")
    print()


def to_json_report(
    results: Sequence[EpisodeResult],
    scorecard: Scorecard,
) -> dict:
    """Return a JSON-serializable dict combining raw results and the scorecard."""
    return {
        "scorecard": asdict(scorecard),
        "results": [asdict(r) for r in results],
    }
