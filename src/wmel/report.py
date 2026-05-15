"""Reporting helpers: print a scorecard, dump a JSON report, render Markdown."""

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
    header = f"Scorecard: {scorecard.policy_name}"
    if scorecard.perturbation_name:
        header += f"  (perturbation: {scorecard.perturbation_name})"
    print(header)
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


def _md_value(value: float | None, decimals: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{decimals}f}"


def to_markdown_scorecard(scorecard: Scorecard) -> str:
    """Render a `Scorecard` as a Markdown table, paste-ready for a PR or doc."""
    heading = f"### Scorecard: `{scorecard.policy_name}`"
    if scorecard.perturbation_name:
        heading += f" (perturbation: `{scorecard.perturbation_name}`)"
    lines = [
        heading,
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| episodes | {scorecard.episodes} |",
        f"| action success rate | {_md_value(scorecard.success_rate)} |",
        f"| average steps to success | {_md_value(scorecard.average_steps_to_success, decimals=1)} |",
        f"| planning latency per call (ms) | {_md_value(scorecard.average_planning_latency_ms)} |",
        f"| perturbation recovery rate | {_md_value(scorecard.perturbation_recovery_rate)} |",
        f"| average compute per decision | {_md_value(scorecard.average_compute_per_decision)} |",
    ]
    for name, value in scorecard.extras.items():
        lines.append(f"| {name} | {_md_value(value)} |")
    return "\n".join(lines) + "\n"


def to_markdown_report(scorecards: Sequence[Scorecard], heading: str | None = None) -> str:
    """Render one or more `Scorecard`s as a single Markdown document.

    Useful for dropping a comparison directly into a pull request body or a
    docs page.
    """
    parts: list[str] = []
    if heading:
        parts.append(f"# {heading}\n")
    for sc in scorecards:
        parts.append(to_markdown_scorecard(sc))
    return "\n".join(parts)
