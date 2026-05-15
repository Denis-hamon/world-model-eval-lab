"""Reporting helpers: print a scorecard, dump a JSON report, render Markdown."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from typing import Sequence

from wmel.metrics import EpisodeResult, Scorecard


# Schema version of the JSON report envelope. Bump on breaking changes; keep
# the envelope additive (consumers should ignore unknown keys).
REPORT_SCHEMA_VERSION = "1.0"


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


def report_envelope_metadata() -> dict:
    """Return the standard top-level versioning fields for any report dict.

    Use it to stamp `schema_version`, `wmel_version`, and `generated_at`
    on the outer wrapper of scripts that bundle multiple per-run envelopes
    (e.g., `examples/maze_toy/run_baseline.py` returns one dict containing
    three `runs`, and the outer dict should be versioned just like each
    inner one).

    Typical usage:

        report = {
            **report_envelope_metadata(),
            "environment": "maze_toy",
            "runs": {...},
        }
    """
    from wmel import __version__ as _wmel_version

    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "wmel_version": _wmel_version,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def to_json_report(
    results: Sequence[EpisodeResult],
    scorecard: Scorecard,
    *,
    extra_metadata: dict | None = None,
) -> dict:
    """Return a JSON-serializable dict combining raw results and the scorecard.

    The envelope is versioned (`schema_version`) and stamped with the wmel
    version that produced it plus a UTC ISO-8601 timestamp. Downstream
    consumers (a future public scoreboard, for instance) can rely on the
    `schema_version` to handle format evolution; bumps will be additive
    whenever possible.

    Pass `extra_metadata` to attach run-level fields (env name, seed, episode
    count, perturbation strategy, anything else worth round-tripping) under
    a top-level `metadata` block. The block is omitted when None.
    """
    from wmel import __version__ as _wmel_version

    envelope: dict = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "wmel_version": _wmel_version,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "scorecard": asdict(scorecard),
        "results": [asdict(r) for r in results],
    }
    if extra_metadata:
        envelope["metadata"] = dict(extra_metadata)
    return envelope


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
